# -*- coding: utf-8 -*- # Ajout pour assurer l'encodage UTF-8

import os
import io
import time # Nécessaire pour pygame.mixer loop
import wave
import json
import pyaudio
import numpy as np
import re
from faster_whisper import WhisperModel
# --- Suppression de Llama ---
from pathlib import Path
import subprocess
from subprocess import PIPE, STDOUT
# import wave # Already imported
import tempfile
import sys
import random
import threading

# --- Ajout pour Argos Translate ---
try:
    import argostranslate.package
    import argostranslate.translate
    ARGOSTRANSLATE_AVAILABLE = True
    print("Argos Translate found.")
except ImportError:
    print("ERROR: 'argostranslate' library is missing.")
    print("Install it with: pip install argostranslate")
    ARGOSTRANSLATE_AVAILABLE = False
    # On arrête si la dépendance principale est manquante
    sys.exit(1)

# --- Importation et initialisation de Pygame pour l'audio ---
try:
    import pygame
    pygame.mixer.init() # Initialiser le mixer audio de pygame
    PYGAME_AVAILABLE = True
    print("Pygame mixer initialized successfully for MP3 playback.")
except ImportError:
    print("Warning: 'pygame' library not found. MP3 playback disabled.")
    print("Install it using: pip install pygame")
    PYGAME_AVAILABLE = False
except Exception as e:
    print(f"Warning: Failed to initialize pygame mixer: {e}. MP3 playback disabled.")
    PYGAME_AVAILABLE = False

# --- La variable PLAYSOUND_AVAILABLE n'est plus utile ---
PLAYSOUND_AVAILABLE = False # Désactiver explicitement playsound

# Try importing winsound, handle import error if not on Windows
try:
    import winsound
except ImportError:
    winsound = None # Set to None if import fails

class VoiceAssistant:
    # ... (garder toutes les définitions de responses et common_responses si nécessaire) ...
    # responses = { ... }
    # common_responses = { ... }

    # --- Messages Modifiés pour une voix anglaise et traduction EN -> FR ---
    # Ces messages seront DITS en ANGLAIS par l'assistant
    welcome_message = "Hello, I am your voice translation assistant. Tell me what you want to translate from English to French."
    no_speech_msg = "I didn't quite catch that. Could you please repeat more clearly in English?"
    empty_transcription_msg = "I couldn't understand what you said in English. Can you please rephrase?"
    translation_error_msg = "Sorry, an error occurred during translation. Please try again."
    # Messages console (gardés pour clarté)
    recording_prompt_msg = "Recording... (Speak English now for translation)"
    transcription_lang_msg = "English" # Pour la clarté de la console sur ce qui est attendu
    translation_direction_msg = "English to French" # Pour la clarté de la console sur l'opération
    assistant_voice_lang_msg = "English" # Pour la clarté de la console sur la voix de l'assistant

    def __init__(self):
        print(f"Initializing voice translator assistant ({self.translation_direction_msg}) with {self.assistant_voice_lang_msg} voice...") # Modifié
        self.pygame_available = PYGAME_AVAILABLE
        self.argostranslate_available = ARGOSTRANSLATE_AVAILABLE

        if not self.argostranslate_available:
            raise RuntimeError("Argos Translate is not available. The assistant cannot start.")

        # --- Configuration des chemins (inchangée) ---
        root_dir = Path.cwd()
        piper_dir = root_dir / "piper"
        models_dir = root_dir / "models" # Pour Piper (TTS)
        sounds_dir = root_dir / "sounds" # Pour d'éventuels sons MP3

        if not sounds_dir.exists():
            print(f"Warning: Sounds directory not found at {sounds_dir}. Creating it.")
            try:
                sounds_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error creating sounds directory: {e}. MP3 playback might fail.")

        self.piper_exe = piper_dir / "piper.exe"

        # --- Chemins des modèles TTS (BESOIN DES DEUX: EN pour la voix, FR pour le résultat traduit) ---
        self.tts_model_fr = models_dir / "fr_FR-upmc-medium.onnx"
        self.tts_config_fr = models_dir / "fr_FR-upmc-medium.onnx.json"
        self.tts_model_en = models_dir / "en_US-amy-medium.onnx" # Nécessaire pour la voix anglaise
        self.tts_config_en = models_dir / "en_US-amy-medium.onnx.json" # Nécessaire pour la voix anglaise

        # --- Vérification des fichiers essentiels (ajout de la vérification EN) ---
        essential_paths = [
            (self.piper_exe, "Piper executable"),
            (self.tts_model_fr, "TTS model (French for translation result)"), # Modifié pour clarté
            (self.tts_config_fr, "TTS config (French for translation result)"), # Modifié pour clarté
            (self.tts_model_en, "TTS model (English for assistant voice)"), # Modifié pour clarté
            (self.tts_config_en, "TTS config (English for assistant voice)"), # Modifié pour clarté
        ]
        for path, desc in essential_paths:
            if not path.exists():
                raise FileNotFoundError(f"{desc} not found at {path}")
            else:
                print(f"Found {desc} at {path}")

        # --- Stockage des chemins en str (inchangés) ---
        self.piper_exe_str = str(self.piper_exe)
        self.tts_model_fr_str = str(self.tts_model_fr)
        self.tts_config_fr_str = str(self.tts_config_fr)
        self.tts_model_en_str = str(self.tts_model_en)
        self.tts_config_en_str = str(self.tts_config_en)

        # --- Configuration audio (inchangée) ---
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.record_seconds = 7
        self.silence_threshold = 0.01
        self.volume_norm = 0.3
        _pa = pyaudio.PyAudio()
        try:
            self.sample_width = _pa.get_sample_size(self.audio_format)
        finally:
            _pa.terminate()

        # --- Dossiers de sortie et temporaire (inchangés) ---
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix='voice_translator_'))
            print(f"Using temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error creating temporary directory: {e}")
            raise

        # --- Initialisation Whisper (inchangée, attend toujours de l'anglais) ---
        print("Initializing Whisper model...")
        try:
            import torch
            self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
             self.device_type = "cpu"
        print(f"Using device: {self.device_type}")
        compute_type = "float16" if self.device_type == "cuda" else "int8" if self.device_type == "cpu" else "float32"
        whisper_model_size = "small" # Vous pouvez utiliser "base", "small", "medium", "large"
        print(f"Loading Whisper model: {whisper_model_size} (compute_type: {compute_type})")
        try:
            # Note: Whisper peut détecter la langue automatiquement, mais la spécifier améliore la précision.
            # Nous la spécifierons dans transcribe_audio() comme 'en'
            cpu_threads = os.cpu_count() // 2 if self.device_type == "cpu" else 0
            self.whisper = WhisperModel(whisper_model_size, device=self.device_type, compute_type=compute_type, cpu_threads=cpu_threads)
            print(f"Whisper model loaded. Using {cpu_threads} CPU threads for transcription if applicable.")
        except Exception as e:
             print(f"Error initializing WhisperModel: {e}")
             raise

        # --- Initialisation Argos Translate (EN -> FR) ---
        print("Initializing Argos Translate engine (English -> French)...")
        self.source_lang_code = "en"
        self.target_lang_code = "fr"

        self.translation_engine = None
        try:
            print("Updating Argos Translate package index...")
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            installed_packages = argostranslate.package.get_installed_packages()
            needed_translation_found = False

            # Vérifier si le package EN -> FR est installé
            for pkg in installed_packages:
                if pkg.from_code == self.source_lang_code and pkg.to_code == self.target_lang_code:
                    needed_translation_found = True
                    print(f"Found installed package: {pkg.from_code} -> {pkg.to_code}")
                    break

            if not needed_translation_found:
                print(f"Direct package {self.source_lang_code} -> {self.target_lang_code} not explicitly installed. Checking availability...")
                package_to_install = next(
                    filter( lambda x: x.from_code == self.source_lang_code and x.to_code == self.target_lang_code, available_packages, ), None
                )
                if package_to_install:
                    print(f"Found direct package {self.source_lang_code} -> {self.target_lang_code}. Attempting download and installation...")
                    try:
                        argostranslate.package.install_from_path(package_to_install.download())
                        print("Package installed successfully.")
                        installed_packages = argostranslate.package.get_installed_packages() # Recharger après installation
                    except Exception as install_e: print(f"Error installing package: {install_e}")
                else:
                    print(f"Could not find a direct package for {self.source_lang_code} to {self.target_lang_code}.")
                    # Vérifier si les modèles EN et FR sont installés séparément
                    en_installed = any(p.from_code == 'en' or p.to_code == 'en' for p in installed_packages)
                    fr_installed = any(p.from_code == 'fr' or p.to_code == 'fr' for p in installed_packages)
                    if not en_installed or not fr_installed:
                        raise RuntimeError(f"Required language models ('{self.source_lang_code}' or '{self.target_lang_code}') not found and direct package unavailable/failed to install. Please install manually (e.g., 'argospm install translate-{self.source_lang_code}_{self.target_lang_code}').")
                    else:
                        print("Warning: Direct en->fr package not found/installed, but 'en' and 'fr' seem available. Translation might rely on intermediate languages if possible.")

            print(f"Loading translation engine ({self.source_lang_code} -> {self.target_lang_code})...")
            installed_languages = argostranslate.translate.get_installed_languages()
            source_lang = next((lang for lang in installed_languages if lang.code == self.source_lang_code), None)
            target_lang = next((lang for lang in installed_languages if lang.code == self.target_lang_code), None)

            if not source_lang: raise RuntimeError(f"Source language '{self.source_lang_code}' model not found among installed languages after check/install attempt.")
            if not target_lang: raise RuntimeError(f"Target language '{self.target_lang_code}' model not found among installed languages after check/install attempt.")

            self.translation_engine = source_lang.get_translation(target_lang)

            if not self.translation_engine: raise RuntimeError(f"Could not get translation engine from '{self.source_lang_code}' to '{self.target_lang_code}'. Check installed package capabilities (e.g., need en_fr package).")

            print(f"Argos Translate engine loaded successfully ({self.source_lang_code} -> {self.target_lang_code}).")

        except Exception as e:
            print(f"Error initializing Argos Translate: {e}")
            print("Please ensure the required language models (English, French, and ideally en_fr) are available and network connection is active for downloads.")
            import traceback; traceback.print_exc()
            raise

        print("Initialization complete!")

    # --- get_random_response (inchangé) ---
    def get_random_response(self, category):
        # Not used in this translation flow, but kept for potential future use
        pass

    # --- normalize_audio (inchangé) ---
    def normalize_audio(self, audio_data):
        try:
            audio_float = audio_data.astype(np.float32) / 32768.0
            max_val = np.max(np.abs(audio_float))
            if max_val > 1e-5:
                norm_factor = self.volume_norm / max_val
                audio_float *= norm_factor
                audio_float = np.clip(audio_float, -1.0, 1.0)
            normalized_audio = (audio_float * 32768.0).astype(np.int16)
            return normalized_audio
        except Exception as e:
            print(f"Error during audio normalization: {e}")
            return audio_data

    # --- record_audio (Utilise le message console anglais) ---
    def record_audio(self):
        audio_instance = pyaudio.PyAudio()
        stream = None
        try:
            stream = audio_instance.open(
                format=self.audio_format, channels=self.channels, rate=self.rate,
                input=True, frames_per_buffer=self.chunk
            )
            print(self.recording_prompt_msg) # Message console anglais
            frames = []
            recorded_chunks = 0
            total_chunks = int(self.rate / self.chunk * self.record_seconds)
            while recorded_chunks < total_chunks:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                    recorded_chunks += 1
                except IOError as e:
                    if hasattr(e, 'errno') and e.errno == pyaudio.paInputOverflowed: print(f"Warning: Audio buffer overflow during recording.")
                    else: print(f"Warning: IOError during recording: {e}")
                    continue
            print("Recording finished.")
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            normalized_audio_data = self.normalize_audio(audio_data)
            audio_float = normalized_audio_data.astype(np.float32) / 32768.0
            rms_level = np.sqrt(np.mean(audio_float**2)) if len(audio_float) > 0 else 0.0
            print(f"Normalized RMS Audio level: {rms_level:.4f}")

            if len(normalized_audio_data) == 0:
                print("Warning: Normalized audio data is empty.")
                return None, 0.0

            return normalized_audio_data, rms_level
        except Exception as e:
            print(f"An error occurred during recording: {e}")
            import traceback; traceback.print_exc()
            return None, 0.0
        finally:
            if stream:
                try:
                    if stream.is_active(): stream.stop_stream()
                    stream.close()
                except Exception as e: print(f"Error closing audio stream: {e}")
            if audio_instance:
                try: audio_instance.terminate()
                except Exception as e: print(f"Error terminating PyAudio: {e}")

    # --- speak_sentence (inchangé, la logique de sélection du modèle est déjà là) ---
    def speak_sentence(self, text, language='en', wait=True): # Default language is now 'en'
        if not text:
            print("Speech skipped: No text provided.")
            return False

        # Vérifier si pygame est disponible si on tente de jouer un MP3 (pas le cas ici, on génère du WAV)
        # if file_path_str.lower().endswith(".mp3") and not self.pygame_available:
        #     print("Warning: Cannot play MP3, pygame is not available.")
        #     return False

        timestamp = int(time.time() * 1000)
        output_wav_path = self.temp_dir / f"response_{timestamp}.wav"
        output_wav_str = str(output_wav_path)

        # Sélectionner le modèle et la configuration en fonction de la langue
        if language == 'en':
            model_path = self.tts_model_en_str
            config_path = self.tts_config_en_str
            print(f"Generating speech (English TTS) for: \"{text[:60]}...\"")
        elif language == 'fr':
            model_path = self.tts_model_fr_str
            config_path = self.tts_config_fr_str
            print(f"Generating speech (French TTS) for: \"{text[:60]}...\"")
        else:
            # Default to English voice if language unknown
            print(f"Warning: Unsupported language '{language}' for TTS. Defaulting to English.")
            model_path = self.tts_model_en_str
            config_path = self.tts_config_en_str
            language = 'en' # Force 'en' for the rest of the logic
            print(f"Generating speech (English TTS) for: \"{text[:60]}...\"")


        try:
            process = subprocess.run(
                [ self.piper_exe_str, "--model", model_path, "--config", config_path, "--output_file", output_wav_str, "--length-scale", "1.0" ],
                input=text.encode('utf-8'),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )

            if process.returncode != 0:
                stderr_output = process.stderr.decode('utf-8', errors='ignore')
                print(f"Piper TTS Error (Return Code: {process.returncode}):\nStderr: {stderr_output}")
                stdout_output = process.stdout.decode('utf-8', errors='ignore')
                if stdout_output: print(f"Piper Stdout: {stdout_output}")
                return False

            if output_wav_path.exists() and output_wav_path.stat().st_size > 1024:
                if wait:
                    self._play_audio_sync(output_wav_str)
                else:
                    thread = threading.Thread(target=self._play_audio_async, args=(output_wav_str,))
                    thread.daemon = True
                    thread.start()
                return True
            else:
                stderr_output = process.stderr.decode('utf-8', errors='ignore'); stdout_output = process.stdout.decode('utf-8', errors='ignore')
                print(f"Piper TTS Error: Output file not created or is too small."); print(f"  Path: {output_wav_str}"); print(f"  Exists: {output_wav_path.exists()}")
                if output_wav_path.exists(): print(f"  Size: {output_wav_path.stat().st_size} bytes")
                if stdout_output: print(f"Piper Stdout: {stdout_output}")
                if stderr_output: print(f"Piper Stderr: {stderr_output}")
                return False

        except FileNotFoundError:
            print(f"Error: Piper executable not found at {self.piper_exe_str}")
            return False
        except Exception as e:
            print(f"Error during speech synthesis or playback: {e}")
            import traceback; traceback.print_exc()
            return False

    # --- _play_audio_sync (inchangé) ---
    def _play_audio_sync(self, wav_file_path):
        try:
            if not Path(wav_file_path).exists(): print(f"Error playing audio: File does not exist: {wav_file_path}"); return
            if sys.platform == 'win32' and winsound: winsound.PlaySound(wav_file_path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            elif sys.platform == 'darwin': subprocess.run(['afplay', wav_file_path], check=True, capture_output=True)
            elif sys.platform.startswith('linux'):
                player = None
                try: subprocess.run(['paplay', '--version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); player = 'paplay'
                except (FileNotFoundError, subprocess.CalledProcessError):
                    try: subprocess.run(['aplay', '--version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); player = 'aplay'
                    except (FileNotFoundError, subprocess.CalledProcessError): print("Warning: No audio playback command (paplay, aplay) found.")
                if player:
                    try: subprocess.run([player, wav_file_path], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    except subprocess.CalledProcessError as e: print(f"Error playing audio with {player}: {e}")
            else: print(f"Warning: Audio playback not implemented for platform: {sys.platform}")
        except Exception as e: print(f"Error playing audio file {wav_file_path}: {e}")
        finally:
            if sys.platform == 'win32': time.sleep(0.1)
            try:
                file_path_obj = Path(wav_file_path);
                if file_path_obj.exists(): file_path_obj.unlink()
            except PermissionError: print(f"Warning: Permission denied removing {wav_file_path}.")
            except Exception as e: print(f"Warning: Could not remove temporary audio file {wav_file_path}: {e}")

    # --- _play_audio_async (inchangé) ---
    def _play_audio_async(self, wav_file_path):
        self._play_audio_sync(wav_file_path)

    # --- transcribe_audio (inchangé, transcrit toujours l'anglais attendu) ---
    def transcribe_audio(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            print("Transcription skipped: No audio data.")
            return ""

        temp_wav_path = self.temp_dir / f'temp_transcribe_{int(time.time()*1000)}.wav'
        temp_wav_str = str(temp_wav_path)

        try:
            with wave.open(temp_wav_str, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.rate)
                wf.writeframes(audio_data.tobytes())

            print(f"Transcribing audio (expecting {self.transcription_lang_msg})...")
            segments, info = self.whisper.transcribe(
                temp_wav_str,
                language="en", # Spécifier l'anglais
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=350)
            )

            transcription = " ".join([segment.text for segment in segments]).strip().lower()
            transcription = re.sub(r'\[.*?\]|\(.*?\)', '', transcription).strip()
            transcription = re.sub(r'\s+', ' ', transcription).strip()

            if transcription:
                print(f"Transcription successful ({self.transcription_lang_msg}): \"{transcription}\"")
            else:
                print("Transcription result is empty after processing.")
            return transcription

        except wave.Error as e:
            print(f"Wave Error during transcription pre-processing: {e}")
            return ""
        except Exception as e:
            print(f"Error during transcription: {e}")
            import traceback; traceback.print_exc()
            return ""
        finally:
            try:
                if temp_wav_path.exists(): temp_wav_path.unlink()
            except Exception as e:
                print(f"Warning: Could not remove temp transcription file {temp_wav_str}: {e}")

    # --- get_response (inchangé, traduit toujours en->fr) ---
    def get_response(self, text):
        if not text:
            print("Translation skipped: No input text.")
            return ""
        print(f"Translating text from {self.source_lang_code} to {self.target_lang_code}...")
        try:
            # Utilise l'engine configuré en EN -> FR dans __init__
            translated_text = self.translation_engine.translate(text)
            if translated_text:
                print("Translation successful.")
                return translated_text
            else:
                print("Translation returned an empty result.")
                # Retourne le *texte* du message d'erreur (défini en anglais dans __init__)
                return self.translation_error_msg
        except Exception as e:
            print(f"Error during translation: {e}")
            import traceback; traceback.print_exc()
            # Retourne le *texte* du message d'erreur (défini en anglais dans __init__)
            return self.translation_error_msg

    # --- Fonction run MODIFIÉE pour parler en ANGLAIS (sauf pour le résultat) ---
    def run(self):
        try:
            # Message d'accueil en anglais
            print(f"\n{'='*30}\n Voice Translator Assistant ({self.translation_direction_msg}) ready.\n Using {self.assistant_voice_lang_msg} voice.\n Press Ctrl+C to stop.\n{'='*30}\n")
            self.speak_sentence(self.welcome_message, language='en', wait=True) # <<< VOIX ANGLAISE

            while True:
                try:
                    # 1. Enregistrer l'audio de l'utilisateur (attendu en anglais)
                    audio_data, audio_level = self.record_audio()

                    if audio_data is None:
                        print("Recording failed.")
                        # Message d'erreur parlé en anglais
                        self.speak_sentence("Sorry, the recording failed. Please try again.", language='en', wait=True) # <<< VOIX ANGLAISE
                        time.sleep(1)
                        continue

                    # Ignorer si trop silencieux
                    if audio_level < self.silence_threshold:
                        print(f"Audio level too low (RMS: {audio_level:.4f} < Threshold: {self.silence_threshold}). Ignoring.")
                        continue

                    # 2. Transcrire l'audio en texte (anglais)
                    transcribed_text = self.transcribe_audio(audio_data)

                    if not transcribed_text:
                        print("Transcription failed or empty.")
                        # Message d'erreur parlé en anglais (utilisant la variable de classe)
                        self.speak_sentence(self.empty_transcription_msg, language='en', wait=True) # <<< VOIX ANGLAISE
                        continue

                    # 3. Traduire le texte (anglais -> français)
                    print(f"Translating ({self.translation_direction_msg}): \"{transcribed_text}\"")
                    translated_text = self.get_response(transcribed_text)

                    # 4. Gérer le résultat de la traduction
                    if translated_text and translated_text != self.translation_error_msg:
                        print(f"-> Translated text ({self.target_lang_code}): \"{translated_text}\"")

                        ### MODIFICATION CLÉ : Parler le texte traduit en FRANÇAIS ###
                        print(f"Speaking the translated text ({self.target_lang_code})...")
                        # Utiliser la langue cible (fr) pour parler le RÉSULTAT
                        self.speak_sentence(translated_text, language=self.target_lang_code, wait=True) # <<< VOIX FRANÇAISE (pour le résultat)

                    elif translated_text == self.translation_error_msg:
                        print("Translation failed.")
                        # Parler le message d'erreur de traduction en anglais (car c'est ce que contient la variable)
                        self.speak_sentence(self.translation_error_msg, language='en', wait=True) # <<< VOIX ANGLAISE
                    else: # Cas où la traduction renvoie une chaîne vide sans être l'erreur spécifique
                        print("Translation returned empty string unexpectedly.")
                         # Parler un message d'erreur générique en anglais
                        self.speak_sentence("I was unable to translate that.", language='en', wait=True) # <<< VOIX ANGLAISE

                    print(f"\nReady for the next command (speak {self.transcription_lang_msg})...") # Message console

                except Exception as e:
                    print(f"\n--- Error in main loop iteration ---")
                    print(f"Error: {e}")
                    import traceback; traceback.print_exc()
                    # Message d'erreur inattendue parlé en anglais
                    error_message = "Oops, an unexpected error occurred. We will try again."
                    print(f"Speaking error message: \"{error_message}\"")
                    try:
                        self.speak_sentence(error_message, language='en', wait=True) # <<< VOIX ANGLAISE
                    except Exception as speak_err:
                        print(f"Could not speak the error message: {speak_err}")
                    print("----------------------------------\n")
                    time.sleep(2) # Pause avant de réessayer
                    continue # Continuer la boucle principale

        except KeyboardInterrupt:
            print("\nCtrl+C detected. Stopping...")
            # Message d'adieu en anglais
            farewell_message = "Translation finished. Goodbye!"
            print(f"Speaking farewell: \"{farewell_message}\"")
            try:
                self.speak_sentence(farewell_message, language='en', wait=True) # <<< VOIX ANGLAISE
            except Exception as e:
                print(f"Could not speak farewell message: {e}")
        finally:
            self.cleanup()

    # --- cleanup (inchangé) ---
    def cleanup(self):
        print("Cleaning up resources...")
        try:
            import shutil
            if self.pygame_available and pygame.mixer.get_init():
                try: pygame.mixer.quit(); print("Pygame mixer quit.")
                except Exception as e: print(f"Error quitting pygame mixer: {e}")
            if hasattr(self, 'temp_dir') and self.temp_dir and self.temp_dir.exists():
                print(f"Removing temporary directory: {self.temp_dir}")
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            print("Cleanup finished.")
        except Exception as e: print(f"Error during cleanup: {e}")

# --- Bloc d'exécution principal (inchangé) ---
if __name__ == "__main__":
    # --- Vérifications initiales (inchangées) ---
    if not ARGOSTRANSLATE_AVAILABLE:
        print("\nFatal Error: Argos Translate library is required but not found.")
        sys.exit(1)

    try:
        # Vérification PyAudio
        if 'pyaudio' not in sys.modules:
           try:
               import pyaudio
               pa = pyaudio.PyAudio()
               pa.terminate()
               print("PyAudio imported and initialized successfully.")
           except ImportError as e:
               print(f"\nFatal Error: PyAudio module not found: {e}")
               print("Please install it (e.g., 'pip install pyaudio') and ensure PortAudio is installed on your system.")
               sys.exit(1)
           except OSError as e:
               print(f"\nFatal Error: PyAudio could not find PortAudio library: {e}")
               print("Please install PortAudio development libraries (e.g., 'sudo apt-get install portaudio19-dev' on Debian/Ubuntu, or download from the website).")
               sys.exit(1)

        # Avertissements optionnels
        if not PYGAME_AVAILABLE: print("\nNote: MP3 playback disabled ('pygame' unavailable or failed to initialize).")
        if sys.platform == 'win32' and winsound is None: print("\nWarning: 'winsound' unavailable on Windows. WAV playback for Piper might fail if 'winsound' cannot be imported.")

        # --- Lancement de l'assistant ---
        assistant = VoiceAssistant()
        assistant.run()

    # --- Gestion des erreurs critiques au démarrage ---
    except FileNotFoundError as e:
        filename = e.filename if hasattr(e, 'filename') else str(e)
        # S'assurer que le message d'erreur mentionne les DEUX modèles TTS
        if "TTS model" in str(e):
             print(f"\nFatal Error: A required TTS model file was not found: {filename}")
             print("Please ensure BOTH the English (e.g., en_US-amy-medium) and French (e.g., fr_FR-upmc-medium) Piper TTS models and their .json config files are present in the 'models' directory.")
        else:
            print(f"\nFatal Error: A required file or directory was not found: {filename}")
            print("Please ensure all necessary files (Piper executable, TTS models FR/EN) are in the correct locations relative to the script.")
        sys.exit(1)
    except ImportError as e:
        print(f"\nFatal Error: A required Python library is missing: {e.name if hasattr(e, 'name') else str(e)}")
        print("Please check your Python environment and install missing packages (e.g., pip install -r requirements.txt if available).")
        sys.exit(1)
    except RuntimeError as e:
         print(f"\nFatal Runtime Error: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"\n--- Critical Error During Startup ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\nProgram finished.")