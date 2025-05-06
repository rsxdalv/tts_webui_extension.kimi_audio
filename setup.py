import setuptools

setuptools.setup(
    name="extension_kimi_audio",
    packages=setuptools.find_namespace_packages(),
    version="0.0.1",
    author="rsxdalv",
    description="Kimi Audio is a powerful text-to-speech and speech-to-text model by Moonshot AI",
    url="https://github.com/rsxdalv/extension_kimi_audio",
    project_urls={},
    scripts=[],
    install_requires=[
        "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp310-cp310-win_amd64.whl ; sys_platform == 'win32'",
        "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp310-cp310-linux_x86_64.whl ; sys_platform == 'linux'",
        "kimia_infer @ https://github.com/rsxdalv/Kimi-Audio/releases/download/v0.1.7/kimia_infer-0.1.7-cp310-cp310-macosx_11_0_universal2.whl ; sys_platform == 'darwin'",
        "soundfile",
        # "transformers>=4.51.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

# PS C:\Users\rob\Desktop\tts-generation-webui-main> python -m workspace.extension_openvoice_v2.extension_openvoice_v2.main
# python -m workspace.extension_kimi_audio.extension_kimi_audio.main
