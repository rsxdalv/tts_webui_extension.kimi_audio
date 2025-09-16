import setuptools

setuptools.setup(
    name="tts_webui_extension.kimi_audio",
    packages=setuptools.find_namespace_packages(),
    version="0.0.1",
    author="rsxdalv",
    description="Kimi Audio is a powerful text-to-speech and speech-to-text model by Moonshot AI",
    url="https://github.com/rsxdalv/tts_webui_extension.kimi_audio",
    project_urls={},
    scripts=[],
    install_requires=[
        "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp310-cp310-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.10'",
        "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp310-cp310-linux_x86_64.whl ; sys_platform == 'linux' and python_version == '3.10'",
        # Mac wheel does not exist yet
        # "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp310-cp310-macosx_11_0_universal2.whl ; sys_platform == 'darwin' and python_version == '3.10'",
        "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp311-cp311-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.11'",
        "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux' and python_version == '3.11'",
        # "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp311-cp311-macosx_11_0_universal2.whl ; sys_platform == 'darwin' and python_version == '3.11'",
        "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp312-cp312-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.12'",
        "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp312-cp312-linux_x86_64.whl ; sys_platform == 'linux' and python_version == '3.12'",
        "soundfile",
        "flash-attn @ https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.10'",
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl ; sys_platform == 'linux' and python_version == '3.10'",
        # "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-macosx_11_0_universal2.whl ; sys_platform == 'darwin' and python_version == '3.10'",
        "flash-attn @ https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp311-cp311-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.11'",
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl ; sys_platform == 'linux' and python_version == '3.11'",
        # "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-macosx_11_0_universal2.whl ; sys_platform == 'darwin' and python_version == '3.11'",
        "flash-attn @ https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp312-cp312-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.12'",
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl ; sys_platform == 'linux' and python_version == '3.12'",
        # "transformers>=4.51.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

