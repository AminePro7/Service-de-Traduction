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
# from llama_cpp import Llama
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
    print("Argos Translate trouvé.")
except ImportError:
    print("ERREUR: La bibliothèque 'argostranslate' est manquante.")
    print("Installez-la avec : pip install argostranslate")
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
    # ... (garder toutes les définitions de responses et common_responses) ...
    responses = {
        # ... (votre code existant pour responses) ...
    }
    common_responses = {
        # ... (votre code existant pour common_responses) ...
    }
    # --- Messages Modifiés ---
    welcome_message = "Bonjour, je suis votre assistant vocal de traduction. Dites-moi ce que vous voulez traduire du français vers l'anglais."
    no_speech_msg = "Je n'ai pas bien entendu. Pourriez-vous répéter plus clairement s'il vous plaît ?"
    empty_transcription_msg = "Je n'ai pas réussi à comprendre ce que vous avez dit. Pouvez-vous reformuler ?"
    translation_error_msg = "Désolé, une erreur est survenue pendant la traduction. Veuillez réessayer."

    def __init__(self):
        print("Initializing voice translator assistant...")
        self.pygame_available = PYGAME_AVAILABLE
        self.argostranslate_available = ARGOSTRANSLATE_AVAILABLE

        if not self.argostranslate_available:
            raise RuntimeError("Argos Translate n'est pas disponible. L'assistant ne peut pas démarrer.")

        root_dir = Path.cwd()
        piper_dir = root_dir / "piper"
        models_dir = root_dir / "models" # Pour Piper (TTS)
        sounds_dir = root_dir / "sounds" # Pour d'éventuels sons MP3

        # --- Création du dossier sounds s'il n'existe pas ---
        if not sounds_dir.exists():
            print(f"Warning: Sounds directory not found at {sounds_dir}. Creating it.")
            try:
                sounds_dir.mkdir(parents=True, exist_ok=True) # Assure la création même si parents manquent
            except Exception as e:
                print(f"Error creating sounds directory: {e}. MP3 playback might fail.")

        self.piper_exe = piper_dir / "piper.exe"

        ### MODIFICATION START: Ajouter les chemins pour les modèles TTS français ET anglais ###
        # Modèle TTS Français (existant)
        self.tts_model_fr = models_dir / "fr_FR-upmc-medium.onnx"
        self.tts_config_fr = models_dir / "fr_FR-upmc-medium.onnx.json"

        # Modèle TTS Anglais (nouveau) - Adaptez les chemins si nécessaire
        self.tts_model_en = models_dir / "en_US-amy-medium.onnx"
        self.tts_config_en = models_dir / "en_US-amy-medium.onnx.json"
        ### MODIFICATION END ###

        # --- Vérification de l'existence des fichiers essentiels ---
        essential_paths = [
            (self.piper_exe, "Piper executable"),
            (self.tts_model_fr, "TTS model (French)"),
            (self.tts_config_fr, "TTS config (French)"),
            ### MODIFICATION START: Ajouter les modèles anglais à la vérification ###
            (self.tts_model_en, "TTS model (English)"),
            (self.tts_config_en, "TTS config (English)"),
            ### MODIFICATION END ###
        ]

        for path, desc in essential_paths:
            if not path.exists():
                raise FileNotFoundError(f"{desc} not found at {path}")
            else:
                print(f"Found {desc} at {path}")

        self.piper_exe_str = str(self.piper_exe)
        ### MODIFICATION START: Stocker les chemins des modèles en str ###
        self.tts_model_fr_str = str(self.tts_model_fr)
        self.tts_config_fr_str = str(self.tts_config_fr)
        self.tts_model_en_str = str(self.tts_model_en)
        self.tts_config_en_str = str(self.tts_config_en)
        ### MODIFICATION END ###


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
            _pa.terminate() # Assurer la fermeture de PyAudio

        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        # Utiliser try-except pour la création du dossier temporaire
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix='voice_translator_'))
            print(f"Using temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error creating temporary directory: {e}")
            raise # Renvoyer l'exception car c'est critique

        print("Initializing Whisper model...")
        try:
            import torch
            self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
             self.device_type = "cpu"
        print(f"Using device: {self.device_type}")
        compute_type = "float16" if self.device_type == "cuda" else "int8" if self.device_type == "cpu" else "float32" # Utiliser int8 sur CPU si possible
        whisper_model_size = "small"
        print(f"Loading Whisper model: {whisper_model_size} (compute_type: {compute_type})")
        try:
            cpu_threads = os.cpu_count() // 2 if self.device_type == "cpu" else 0
            self.whisper = WhisperModel(whisper_model_size, device=self.device_type, compute_type=compute_type, cpu_threads=cpu_threads)
            print(f"Whisper model loaded. Using {cpu_threads} CPU threads for transcription if applicable.")
        except Exception as e:
             print(f"Error initializing WhisperModel: {e}")
             raise

        # --- Initialisation Argos Translate (INCHANGÉE) ---
        print("Initializing Argos Translate engine...")
        self.source_lang_code = "fr"
        self.target_lang_code = "en"
        self.translation_engine = None
        try:
            print("Updating Argos Translate package index...")
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            installed_packages = argostranslate.package.get_installed_packages()
            needed_translation_found = False
            for pkg in installed_packages:
                if pkg.from_code == self.source_lang_code and pkg.to_code == self.target_lang_code:
                    needed_translation_found = True
                    break
                if pkg.from_code == self.source_lang_code or pkg.to_code == self.source_lang_code: pass
                if pkg.from_code == self.target_lang_code or pkg.to_code == self.target_lang_code: pass

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
                        installed_packages = argostranslate.package.get_installed_packages()
                    except Exception as install_e: print(f"Error installing package: {install_e}")
                else:
                    print(f"Could not find a direct package for {self.source_lang_code} to {self.target_lang_code}.")
                    fr_installed = any(p.from_code == 'fr' or p.to_code == 'fr' for p in installed_packages)
                    en_installed = any(p.from_code == 'en' or p.to_code == 'en' for p in installed_packages)
                    if not fr_installed or not en_installed: raise RuntimeError(f"Required language models ('{self.source_lang_code}' or '{self.target_lang_code}') not found and direct package unavailable/failed to install. Please install manually.")
                    else: print("Warning: Direct fr->en package not found/installed, but 'fr' and 'en' seem available. Translation might rely on intermediate languages if possible.")

            print(f"Loading translation engine ({self.source_lang_code} -> {self.target_lang_code})...")
            installed_languages = argostranslate.translate.get_installed_languages()
            source_lang = next((lang for lang in installed_languages if lang.code == self.source_lang_code), None)
            target_lang = next((lang for lang in installed_languages if lang.code == self.target_lang_code), None)

            if not source_lang: raise RuntimeError(f"Source language '{self.source_lang_code}' model not found among installed languages after check/install attempt.")
            if not target_lang: raise RuntimeError(f"Target language '{self.target_lang_code}' model not found among installed languages after check/install attempt.")

            self.translation_engine = source_lang.get_translation(target_lang)

            if not self.translation_engine: raise RuntimeError(f"Could not get translation engine from '{self.source_lang_code}' to '{self.target_lang_code}'. Check installed package capabilities (e.g., need fr_en package).")

            print(f"Argos Translate engine loaded successfully ({self.source_lang_code} -> {self.target_lang_code}).")

        except Exception as e:
            print(f"Error initializing Argos Translate: {e}")
            print("Please ensure the required language models are available (run the manual install script if needed) and network connection is active for downloads.")
            import traceback; traceback.print_exc()
            raise

        print("Initialization complete!")

    # ... (get_random_response, normalize_audio, record_audio, _play_audio_sync, _play_audio_async, transcribe_audio - méthodes inchangées) ...

    def get_random_response(self, category):
        if category in self.responses and self.responses[category]:
            return random.choice(self.responses[category])
        print(f"Warning: Response category '{category}' not found or empty.")
        return "" # Retourner une chaîne vide si la catégorie n'existe pas

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
            return audio_data # Retourner les données originales en cas d'erreur

    def record_audio(self):
        audio_instance = pyaudio.PyAudio()
        stream = None
        try:
            stream = audio_instance.open(
                format=self.audio_format, channels=self.channels, rate=self.rate,
                input=True, frames_per_buffer=self.chunk
            )
            print("Recording... (Parlez maintenant pour la traduction)")
            frames = []
            recorded_chunks = 0
            total_chunks = int(self.rate / self.chunk * self.record_seconds)
            while recorded_chunks < total_chunks:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                    recorded_chunks += 1
                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed: print(f"Warning: Audio buffer overflow during recording.")
                    else: print(f"Warning: IOError during recording: {e}")
                    continue
            print("Recording finished.")
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            normalized_audio_data = self.normalize_audio(audio_data)
            audio_float = normalized_audio_data.astype(np.float32) / 32768.0
            rms_level = np.sqrt(np.mean(audio_float**2))
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

    ### MODIFICATION START: Modifier speak_sentence pour accepter une langue ###
    def speak_sentence(self, text, language='fr', wait=True):
        """
        Génère et joue un fichier audio à partir du texte en utilisant Piper TTS.
        Peut choisir le modèle de langue (fr ou en).

        Args:
            text (str): Le texte à synthétiser.
            language (str): Le code de langue ('fr' ou 'en'). Défaut 'fr'.
            wait (bool): Si True, attend la fin de la lecture. Si False, joue en arrière-plan.
        """
        if not text:
            print("Speak request ignored: No text provided.")
            return False
        if sys.platform == 'win32' and winsound is None:
             print("Warning: winsound might be required for audio playback on Windows for Piper's WAV output.")
             # Ne pas retourner False ici, la gestion se fera dans _play_audio_sync

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
            print(f"Warning: Unsupported language '{language}' for TTS. Defaulting to French.")
            model_path = self.tts_model_fr_str
            config_path = self.tts_config_fr_str
            language = 'fr' # Force 'fr' pour le reste de la logique

        try:
            # Utiliser les variables model_path et config_path sélectionnées
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
    ### MODIFICATION END ###


    def _play_audio_sync(self, wav_file_path):
        # ... (code inchangé) ...
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


    def _play_audio_async(self, wav_file_path):
        self._play_audio_sync(wav_file_path)


    def transcribe_audio(self, audio_data):
        # ... (code inchangé) ...
        if audio_data is None or len(audio_data) == 0: print("Transcription skipped: No audio data."); return ""
        temp_wav_path = self.temp_dir / f'temp_transcribe_{int(time.time()*1000)}.wav'
        temp_wav_str = str(temp_wav_path)
        try:
            with wave.open(temp_wav_str, 'wb') as wf:
                wf.setnchannels(self.channels); wf.setsampwidth(self.sample_width); wf.setframerate(self.rate); wf.writeframes(audio_data.tobytes())

            segments, info = self.whisper.transcribe(
                temp_wav_str, language="fr", beam_size=5,
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=350)
            )

            transcription = " ".join([segment.text for segment in segments]).strip().lower()
            transcription = re.sub(r'\[.*?\]|\(.*?\)', '', transcription).strip()
            transcription = re.sub(r'\s+', ' ', transcription).strip()

            if transcription: print(f"Transcription successful: \"{transcription}\"")
            else: print("Transcription result is empty after processing.")
            return transcription

        except wave.Error as e: print(f"Wave Error during transcription pre-processing: {e}"); return ""
        except Exception as e: print(f"Error during transcription: {e}"); import traceback; traceback.print_exc(); return ""
        finally:
            try:
                if temp_wav_path.exists(): temp_wav_path.unlink()
            except Exception as e: print(f"Warning: Could not remove temp transcription file {temp_wav_str}: {e}")


    # --- Fonction get_response (INCHANGÉE, fait juste la traduction) ---
    def get_response(self, text):
        if not text:
            print("Translation skipped: No input text.")
            return ""
        print(f"Translating text from {self.source_lang_code} to {self.target_lang_code}...")
        try:
            translated_text = self.translation_engine.translate(text)
            if translated_text:
                print("Translation successful.")
                return translated_text
            else:
                print("Translation returned an empty result.")
                return self.translation_error_msg
        except Exception as e:
            print(f"Error during translation: {e}")
            import traceback; traceback.print_exc()
            return self.translation_error_msg

    # --- Fonction run MODIFIÉE pour utiliser la traduction ET parler en anglais ---
    def run(self):
        try:
            print("\n" + "="*30 + "\n Voice Translator Assistant is ready.\n Press Ctrl+C to stop.\n" + "="*30 + "\n")
            # Message d'accueil toujours en français
            self.speak_sentence(self.welcome_message, language='fr', wait=True)

            while True:
                try:
                    # 1. Enregistrer l'audio de l'utilisateur (français)
                    audio_data, audio_level = self.record_audio()

                    if audio_data is None:
                        print("Recording failed.")
                        # Message d'erreur en français
                        self.speak_sentence("Désolé, l'enregistrement a échoué. Veuillez réessayer.", language='fr', wait=True)
                        time.sleep(1)
                        continue

                    if audio_level < self.silence_threshold:
                        print(f"Audio level too low (RMS: {audio_level:.4f} < Threshold: {self.silence_threshold}). Ignoring.")
                        continue

                    # 2. Transcrire l'audio en texte (français)
                    print("Transcribing recorded audio...")
                    transcribed_text = self.transcribe_audio(audio_data)

                    if not transcribed_text:
                        print("Transcription failed or empty.")
                        # Message d'erreur en français
                        self.speak_sentence(self.empty_transcription_msg, language='fr', wait=True)
                        continue

                    # 3. Traduire le texte (français -> anglais)
                    print(f"Translating text: \"{transcribed_text}\"")
                    translated_text = self.get_response(transcribed_text)

                    # 4. Gérer le résultat de la traduction
                    if translated_text and translated_text != self.translation_error_msg:
                        print(f"-> Translated text ({self.target_lang_code}): \"{translated_text}\"")

                        ### MODIFICATION START: Parler le texte traduit en anglais ###
                        print(f"Speaking the translated text ({self.target_lang_code})...")
                        # Appeler speak_sentence avec le texte traduit et la langue anglaise ('en')
                        self.speak_sentence(translated_text, language=self.target_lang_code, wait=True)
                        ### MODIFICATION END ###

                    elif translated_text == self.translation_error_msg:
                        print("Translation failed.")
                        # Parler le message d'erreur de traduction en français
                        self.speak_sentence(self.translation_error_msg, language='fr', wait=True)
                    else:
                        print("Translation returned empty string.")
                         # Parler le message d'erreur générique en français
                        self.speak_sentence("Je n'ai pas pu traduire cela.", language='fr', wait=True)

                    print("\nReady for next command...")

                except Exception as e:
                    print(f"\n--- Error in main loop iteration ---")
                    print(f"Error: {e}")
                    import traceback; traceback.print_exc()
                    error_message = "Oups, une erreur inattendue s'est produite. Nous allons réessayer."
                    print(f"Speaking error message: \"{error_message}\"")
                    try:
                         # Message d'erreur inattendue en français
                        self.speak_sentence(error_message, language='fr', wait=True)
                    except Exception as speak_err:
                        print(f"Could not speak the error message: {speak_err}")
                    print("----------------------------------\n")
                    time.sleep(2)
                    continue

        except KeyboardInterrupt:
            print("\nCtrl+C detected. Stopping...")
            farewell_message = "Traduction terminée. Au revoir !"
            print(f"Speaking farewell: \"{farewell_message}\"")
            try:
                # Message d'adieu en français
                self.speak_sentence(farewell_message, language='fr', wait=True)
            except Exception as e:
                print(f"Could not speak farewell message: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        # ... (code inchangé) ...
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

# --- Bloc d'exécution principal (INCHANGÉ) ---
if __name__ == "__main__":
    # ... (votre code existant pour les vérifications et l'exécution) ...
    if not ARGOSTRANSLATE_AVAILABLE:
        print("\nFatal Error: Argos Translate library is required but not found.")
        sys.exit(1)

    try:
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

        if not PYGAME_AVAILABLE: print("\nNote: MP3 playback disabled ('pygame' unavailable or failed to initialize).")
        if sys.platform == 'win32' and winsound is None: print("\nWarning: 'winsound' unavailable on Windows. WAV playback for Piper might fail if 'winsound' cannot be imported.")

        assistant = VoiceAssistant()
        assistant.run()

    except FileNotFoundError as e:
        print(f"\nFatal Error: A required file or directory was not found: {e.filename if hasattr(e, 'filename') else e}") # Affichage amélioré
        print("Please ensure all necessary files (Piper, TTS models FR/EN) are in the correct locations relative to the script.")
        sys.exit(1)
    except ImportError as e:
        print(f"\nFatal Error: A required Python library is missing: {e}")
        print("Please check your Python environment and install missing packages.")
        sys.exit(1)
    except RuntimeError as e:
         print(f"\nFatal Runtime Error: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"\n--- Critical Error ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\nProgram finished.")