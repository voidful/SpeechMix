from setuptools import setup, find_packages

setup(
    name='speechmix',
    version='0.0.26',
    description='Explore different way to mix speech model(wav2vec2, hubert) and nlp model(BART,T5,GPT) together',
    url='https://github.com/voidful/SpeechMix',
    author='Voidful',
    author_email='voidful.stack@gmail.com',
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    setup_requires=['setuptools-git'],
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.6'
    ],
    license="Apache",
    keywords='transformer huggingface nlp speech',
    packages=find_packages(),
    install_requires=[
        "transformers"
    ],
    entry_points={
    },
    python_requires=">=3.5.0",
    zip_safe=False,
)
