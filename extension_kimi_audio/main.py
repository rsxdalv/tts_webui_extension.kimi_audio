import gradio as gr
from typing import Dict, Any

from tts_webui.decorators.decorator_add_base_filename import decorator_add_base_filename
from tts_webui.decorators.decorator_add_date import decorator_add_date
from tts_webui.decorators.decorator_add_model_type import decorator_add_model_type
from tts_webui.decorators.decorator_apply_torch_seed import decorator_apply_torch_seed
from tts_webui.decorators.decorator_log_generation import decorator_log_generation
from tts_webui.decorators.decorator_save_metadata import decorator_save_metadata
from tts_webui.decorators.decorator_save_wav import decorator_save_wav
from tts_webui.decorators.gradio_dict_decorator import dictionarize
from tts_webui.decorators.log_function_time import log_function_time
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
)
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.randomize_seed import randomize_seed_ui


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


@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("kimi_audio")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def generate_speech(
    audio: str,
    text: str,
    model_path: str = "data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
    audio_temperature: float = 0.8,
    audio_top_k: int = 10,
    text_temperature: float = 0.0,
    text_top_k: int = 5,
    audio_repetition_penalty: float = 1.0,
    audio_repetition_window_size: int = 64,
    text_repetition_penalty: float = 1.0,
    text_repetition_window_size: int = 16,
    **kwargs,
) -> Dict[str, Any]:
    """Generate speech from text using Kimi Audio."""
    model = get_kimi_audio_model(model_path, load_detokenizer=True)

    sampling_params = {
        "audio_temperature": audio_temperature,
        "audio_top_k": audio_top_k,
        "text_temperature": text_temperature,
        "text_top_k": text_top_k,
        "audio_repetition_penalty": audio_repetition_penalty,
        "audio_repetition_window_size": audio_repetition_window_size,
        "text_repetition_penalty": text_repetition_penalty,
        "text_repetition_window_size": text_repetition_window_size,
    }

    # {"role": "user", "message_type": "text", "content": text},
    # {"role": "user", "message_type": "audio", "content": audio},
    if audio is not None:
        message = {"role": "user", "message_type": "audio", "content": audio}
    else:
        # this cannot run
        gr.Info("Cannot be run without audio")
        message = {"role": "user", "message_type": "text", "content": text}

    messages_conversation = [message]

    # Generate both audio and text output
    wav_output, text_output = model.generate(
        messages_conversation, **sampling_params, output_type="both"
    )

    sample_rate = 24_000

    return {
        "audio_out": (sample_rate, wav_output.detach().cpu().view(-1).numpy()),
        "text_out": text_output,
    }


def audio_to_text(
    audio: str,
    model_path: str = "data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
    audio_temperature: float = 0.8,
    audio_top_k: int = 10,
    text_temperature: float = 0.0,
    text_top_k: int = 5,
    audio_repetition_penalty: float = 1.0,
    audio_repetition_window_size: int = 64,
    text_repetition_penalty: float = 1.0,
    text_repetition_window_size: int = 16,
    **kwargs,
):
    model = get_kimi_audio_model(model_path)

    sampling_params = {
        "audio_temperature": audio_temperature,
        "audio_top_k": audio_top_k,
        "text_temperature": text_temperature,
        "text_top_k": text_top_k,
        "audio_repetition_penalty": audio_repetition_penalty,
        "audio_repetition_window_size": audio_repetition_window_size,
        "text_repetition_penalty": text_repetition_penalty,
        "text_repetition_window_size": text_repetition_window_size,
    }

    messages_asr = [
        {
            "role": "user",
            "message_type": "text",
            "content": "Please transcribe the following audio:",
        },
        {"role": "user", "message_type": "audio", "content": audio},
    ]

    _, text_output = model.generate(messages_asr, **sampling_params, output_type="text")

    return {"text_out": text_output}


def audio_to_text_conversation(
    audio: str,
    model_path: str = "data/models/kimi-audio/moonshotai/Kimi-Audio-7B-Instruct",
    audio_temperature: float = 0.8,
    audio_top_k: int = 10,
    text_temperature: float = 0.0,
    text_top_k: int = 5,
    audio_repetition_penalty: float = 1.0,
    audio_repetition_window_size: int = 64,
    text_repetition_penalty: float = 1.0,
    text_repetition_window_size: int = 16,
    **kwargs,
):
    model = get_kimi_audio_model(model_path)

    sampling_params = {
        "audio_temperature": audio_temperature,
        "audio_top_k": audio_top_k,
        "text_temperature": text_temperature,
        "text_top_k": text_top_k,
        "audio_repetition_penalty": audio_repetition_penalty,
        "audio_repetition_window_size": audio_repetition_window_size,
        "text_repetition_penalty": text_repetition_penalty,
        "text_repetition_window_size": text_repetition_window_size,
    }

    messages_conversation = [
        {"role": "user", "message_type": "audio", "content": audio}
    ]

    _, text_output = model.generate(
        messages_conversation, **sampling_params, output_type="text"
    )

    return {"text_out": text_output}


def kimi_audio_tts_ui():
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text Input",
                placeholder="Enter text to convert to speech...",
                lines=5,
                visible=False,
            )
            audio_input = gr.Audio(
                label="Audio Input", type="filepath", sources="upload"
            )

            with gr.Row():
                audio_temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.8, step=0.1, label="Audio Temperature"
                )
                text_temperature = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.0, step=0.1, label="Text Temperature"
                )
                audio_top_k = gr.Slider(
                    minimum=1, maximum=100, value=10, step=1, label="Audio Top K"
                )
                text_top_k = gr.Slider(
                    minimum=1, maximum=100, value=5, step=1, label="Text Top K"
                )
                audio_repetition_penalty = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Audio Repetition Penalty"
                )
                audio_repetition_window_size = gr.Slider(
                    minimum=1, maximum=128, value=64, step=1, label="Audio Repetition Window Size"
                )
                text_repetition_penalty = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Text Repetition Penalty"
                )
                text_repetition_window_size = gr.Slider(
                    minimum=1, maximum=128, value=16, step=1, label="Text Repetition Window Size"
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
            text_output = gr.Textbox(label="Generated Text", lines=5)

            seed, randomize_seed_callback = randomize_seed_ui()

    inputs_dict = {
        text_input: "text",
        audio_input: "audio",
        model_path: "model_path",
        audio_temperature: "audio_temperature",
        audio_top_k: "audio_top_k",
        text_temperature: "text_temperature",
        text_top_k: "text_top_k",
        audio_repetition_penalty: "audio_repetition_penalty",
        audio_repetition_window_size: "audio_repetition_window_size",
        text_repetition_penalty: "text_repetition_penalty",
        text_repetition_window_size: "text_repetition_window_size",
        seed: "seed",
    }

    outputs_dict = {
        "audio_out": audio_output,
        "text_out": text_output,
        "metadata": gr.JSON(visible=False),
        "folder_root": gr.Textbox(visible=False),
    }

    generate_btn.click(
        **randomize_seed_callback,
    ).then(**dictionarize(fn=generate_speech, inputs=inputs_dict, outputs=outputs_dict))


def kimi_audio_stt_ui():
    """UI for Kimi Audio speech-to-text."""
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Audio Input", type="filepath", sources="upload"
            )

            with gr.Row():
                audio_temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.8, step=0.1, label="Audio Temperature"
                )
                text_temperature = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.0, step=0.1, label="Text Temperature"
                )
                audio_top_k = gr.Slider(
                    minimum=1, maximum=100, value=10, step=1, label="Audio Top K"
                )
                text_top_k = gr.Slider(
                    minimum=1, maximum=100, value=5, step=1, label="Text Top K"
                )
                audio_repetition_penalty = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Audio Repetition Penalty"
                )
                audio_repetition_window_size = gr.Slider(
                    minimum=1, maximum=128, value=64, step=1, label="Audio Repetition Window Size"
                )
                text_repetition_penalty = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Text Repetition Penalty"
                )
                text_repetition_window_size = gr.Slider(
                    minimum=1, maximum=128, value=16, step=1, label="Text Repetition Window Size"
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

            seed, randomize_seed_callback = randomize_seed_ui()

    common_inputs = {
        audio_input: "audio",
        model_path: "model_path",
        audio_temperature: "audio_temperature",
        audio_top_k: "audio_top_k",
        text_temperature: "text_temperature",
        text_top_k: "text_top_k",
        audio_repetition_penalty: "audio_repetition_penalty",
        audio_repetition_window_size: "audio_repetition_window_size",
        text_repetition_penalty: "text_repetition_penalty",
        text_repetition_window_size: "text_repetition_window_size",
        seed: "seed",
    }

    transcribe_btn.click(**randomize_seed_callback).then(
        **dictionarize(
            fn=audio_to_text,
            inputs=common_inputs,
            outputs={"text_out": text_output},
        )
    )
    conversation_btn.click(**randomize_seed_callback).then(
        **dictionarize(
            fn=audio_to_text_conversation,
            inputs=common_inputs,
            outputs={"text_out": text_output},
        )
    )

def kimi_audio_tab():
    gr.Markdown(
        """
    # Kimi Audio
    
    Kimi Audio is a powerful text-to-speech and speech-to-text model by Moonshot AI.
    """
    )

    with gr.Tabs():
        with gr.Tab("Conversation-Based Speech generation"):
            kimi_audio_tts_ui()

        with gr.Tab("Conversation based Speech-to-Text"):
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
                ## Download Model
                
                Click the button below to download the Kimi Audio model.
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
