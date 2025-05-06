import os
import torch
import gradio as gr
import soundfile as sf
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import tempfile

from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.history_tab.save_to_favorites import save_to_favorites
from tts_webui.decorators.gradio_dict_decorator import dictionarize
from tts_webui.decorators.decorator_save_wav import decorator_save_wav
from tts_webui.decorators.decorator_add_base_filename import decorator_add_base_filename
from tts_webui.decorators.decorator_add_date import decorator_add_date
from tts_webui.decorators.decorator_add_model_type import decorator_add_model_type
from tts_webui.decorators.log_function_time import log_function_time
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_outer,
    decorator_extension_inner,
)


def download_model(
    model_path: str = "data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
):
    from huggingface_hub import snapshot_download

    snapshot_download(
        "moonshotai/Kimi-Audio-7B-Instruct",
        repo_type="model",
        local_dir=model_path,
        local_dir_use_symlinks=False,
    )

    return "ok"


@manage_model_state("kimi_audio")
def get_kimi_audio_model(
    model_path: str = "data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
    load_detokenizer: bool = False,
):
    from kimia_infer.api.kimia import KimiAudio

    return KimiAudio(model_path=model_path, load_detokenizer=load_detokenizer)


# @decorator_extension_outer
# @decorator_apply_torch_seed
# @decorator_save_metadata
# @decorator_save_wav
# @decorator_add_model_type("kimi_audio")
# @decorator_add_base_filename
# @decorator_add_date
# @decorator_log_generation
# @decorator_extension_inner
# @log_function_time
def generate_speech(
    text: str,
    model_path: str = "data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    **kwargs
) -> Dict[str, Any]:
    """Generate speech from text using Kimi Audio."""
    model = get_kimi_audio_model(model_path, load_detokenizer=True)

    # Generate speech
    wav_output, sample_rate = model.generate_speech(
        text=text, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens
    )

    # Convert to numpy array
    if isinstance(wav_output, torch.Tensor):
        wav_output = wav_output.detach().cpu().numpy()

    # Ensure the audio is in the right format
    if len(wav_output.shape) > 1 and wav_output.shape[0] == 1:
        wav_output = wav_output.squeeze(0)

    return {"audio_out": (sample_rate, wav_output)}


def audio_to_text(
    audio: str,
    model_path: str = "data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    **kwargs
):
    model = get_kimi_audio_model(model_path)

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }

    # --- 3. Example 1: Audio-to-Text (ASR) ---
    messages_asr = [
        # You can provide context or instructions as text
        {
            "role": "user",
            "message_type": "text",
            "content": "Please transcribe the following audio:",
        },
        # Provide the audio file path
        {"role": "user", "message_type": "audio", "content": audio},
    ]

    # Generate only text output
    _, text_output = model.generate(messages_asr, **sampling_params, output_type="text")
    print(
        ">>> ASR Output Text: ", text_output
    )  # Expected output: "这并不是告别，这是一个篇章的结束，也是新篇章的开始。"

    return {"text_out": text_output}


def audio_to_text_conversation(
    audio: str,
    model_path: str = "data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    **kwargs
):
    model = get_kimi_audio_model(model_path)

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    # --- 4. Example 2: Audio-to-Audio/Text Conversation ---
    messages_conversation = [
        # Start conversation with an audio query
        # {"role": "user", "message_type": "audio", "content": "test_audios/qa_example.wav"}
        {"role": "user", "message_type": "audio", "content": audio}
    ]

    # Generate both audio and text output
    # wav_output, text_output = model.generate(messages_conversation, **sampling_params, output_type="both")
    _, text_output = model.generate(
        messages_conversation, **sampling_params, output_type="text"
    )

    # Save the generated audio
    # output_audio_path = "output_audio.wav"
    # sf.write(output_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000) # Assuming 24kHz output
    # print(f">>> Conversational Output Audio saved to: {output_audio_path}")
    print(">>> Conversational Output Text: ", text_output)  # Expected output: "A."

    print("Kimi-Audio inference examples complete.")

    # return {"text_out": text_output, "audio_out": (24000, wav_output.detach().cpu().view(-1).numpy())}
    return {"text_out": text_output}


# @decorator_extension_outer
# # @decorator_save_wav
# @decorator_add_model_type("kimi_audio")
# @decorator_add_base_filename
# @decorator_add_date
# @decorator_extension_inner
# @log_function_time
def transcribe_audio(
    audio: str,
    model_path: str = "data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    **kwargs
) -> Dict[str, Any]:
    """Transcribe audio to text using Kimi Audio."""
    model = get_kimi_audio_model(model_path)

    # Load audio file
    audio_array, sample_rate = sf.read(audio)

    # Transcribe audio
    transcription = model.transcribe_audio(
        audio=audio_array,
        sr=sample_rate,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    return {"text_out": transcription, "audio_in": (sample_rate, audio_array)}


def kimi_audio_tts_ui():
    """UI for Kimi Audio text-to-speech."""
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text Input",
                placeholder="Enter text to convert to speech...",
                lines=5,
            )

            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top P"
                )
                max_new_tokens = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=1024,
                    step=128,
                    label="Max New Tokens",
                )

            model_path = gr.Textbox(
                label="Model Path",
                value="data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
                lines=1,
            )

            unload_model_button("kimi_audio")

            generate_btn = gr.Button("Generate Speech", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech", type="numpy")
            with gr.Row():
                folder_root = gr.Textbox(visible=False)
                save_button = gr.Button("Save to favorites", visible=True)

            save_button.click(
                fn=save_to_favorites,
                inputs=[folder_root],
                outputs=[save_button],
            )

    inputs_dict = {
        text_input: "text",
        model_path: "model_path",
        temperature: "temperature",
        top_p: "top_p",
        max_new_tokens: "max_new_tokens",
    }

    outputs_dict = {"audio_out": audio_output, "folder_root": folder_root}

    generate_btn.click(
        **dictionarize(fn=generate_speech, inputs=inputs_dict, outputs=outputs_dict)
    )


def kimi_audio_stt_ui():
    """UI for Kimi Audio speech-to-text."""
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Audio Input", type="filepath", sources="upload"
            )

            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top P"
                )
                max_new_tokens = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=1024,
                    step=128,
                    label="Max New Tokens",
                )

            model_path = gr.Textbox(
                label="Model Path",
                value="data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
                lines=1,
            )

            unload_model_button("kimi_audio")

            transcribe_btn = gr.Button("Transcribe Audio", variant="primary")
            conversation_btn = gr.Button(
                "Transcript to Conversation", variant="primary"
            )

        with gr.Column():
            text_output = gr.Textbox(label="Transcription", lines=5)

    transcribe_btn.click(
        **dictionarize(
            # fn=transcribe_audio,
            fn=audio_to_text,
            inputs={
                audio_input: "audio",
                model_path: "model_path",
                temperature: "temperature",
                top_p: "top_p",
                max_new_tokens: "max_new_tokens",
            },
            outputs={"text_out": text_output},
        )
    )
    conversation_btn.click(
        **dictionarize(
            fn=audio_to_text_conversation,
            inputs={
                audio_input: "audio",
                model_path: "model_path",
                temperature: "temperature",
                top_p: "top_p",
                max_new_tokens: "max_new_tokens",
            },
            outputs={"text_out": text_output},
        )
    )


def kimi_audio_tab():
    """Main tab for Kimi Audio."""
    with gr.Tab("Kimi Audio"):
        gr.Markdown(
            """
        # Kimi Audio
        
        Kimi Audio is a powerful text-to-speech and speech-to-text model by Moonshot AI.
        
        """
        )

        with gr.Tabs():
            # with gr.Tab("Text-to-Speech"):
            #     kimi_audio_tts_ui()

            with gr.Tab("Speech-to-Text"):
                kimi_audio_stt_ui()

            with gr.Tab("Info"):
                gr.Markdown(
                    """
                
                    ## Requirements
                    
                    Download tab might not work, in that case:
                    
                    - Download the model from Hugging Face: [moonshotai/Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct)
                    - Place it in `data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct`
                    - MoonshotAI's Whisper checkpoint might fail, in that case download the checkpoint from [here](https://huggingface.co/openai/whisper-large-v3/blob/main/model.safetensors)
                    
                    ## Features
                    
                    - Text-to-Speech: Convert text to natural-sounding speech
                    - Speech-to-Text: Transcribe audio to text

                    ## Model Sizes and VRAM:

                    Detokenizer | Size on Disk | VRAM Usage |
                    --- | --- | --- |
                    Without Detokenizer | 27.7 GB | 22+ GB |
                    With Detokenizer | 35+ GB | 30? GB |

                """
                )

            with gr.Tab("Download"):
                gr.Markdown(
                    """
                    ## Download the model from Hugging Face: [moonshotai/Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct)
                    ## Place it in `data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct`
                    """
                )

                gr.Button("Download Model", variant="primary").click(
                    fn=download_model,
                    inputs=[],
                    outputs=[gr.Markdown()],
                )


def ui():
    """Main UI function."""
    kimi_audio_tab()


def extension__tts_generation_webui():
    """Extension entry point."""
    with gr.Tab("Kimi Audio"):
        ui()

    return {
        "package_name": "extension_kimi_audio",
        "name": "Kimi Audio",
        "version": "0.0.1",
        "requirements": "git+https://github.com/rsxdalv/extension_kimi_audio@main",
        "description": "Kimi Audio is a powerful text-to-speech and speech-to-text model by Moonshot AI",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "Moonshot AI",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/moonshotai/Kimi-Audio",
        "extension_website": "https://github.com/rsxdalv/extension_kimi_audio",
        "extension_platform_version": "0.0.1",
    }


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()  # type: ignore
    with gr.Blocks() as demo:
        with gr.Tab("Kimi Audio"):
            ui()

    demo.launch(
        server_port=7771,
    )
