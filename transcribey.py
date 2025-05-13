# Architecture: 

# 1.) Identify language

model_comparison = [
    {
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "languages": ["English"],
        "WER": 6.05,
        "RTFx": 3386.02,
        "size_gb": 2.47,
        "note": "Most accurate english model"
    },
    {
        "model": "microsoft/Phi-4-multimodal-instruct",
        "languages": ["English", "Chinese", "German", "French", "Italian", "Japanese", "Spanish", "Portuguese"],
        "WER": 6.14,
        "RTFx": 62.12,
        "size_gb": 11.2,
        "note": "Most accurate Chinese, German, French, Italian, Japanese, Spanish, Portuguese Model"
    },
    {
        "model": "nvidia/parakeet-tdt_ctc-110m",
        "languages": ["English"],
        "WER": 7.49,
        "RTFx": 5345.14,
        "size_gb": 0.5,
        "note": "Fastest English Model"
    },
    {
        "model": "nvidia/canary-1b-flash",
        "languages": ["English", "German", "French", "Spanish"],
        "WER": 6.35,
        "RTFx": 1045.75,
        "size_gb": 3.54,
        "note": "Fastest German, French, and Spanish Model"
    }
]

# We have a ton of files that need transcribing. 

# We also have a ton of computers that have numerous GPUs.

# So, how do we do this the best? 

# We have access to say a folder, with a number of files on it.

# VCons will be saved in a folder as well. 

# Looks like the conserver can do all this. 

# So, run a conserver? 

# I don't think the conserver will be fast enough. 

# To that end, 

# We need to identify language and if english then we use fastest english model. 


# fastest english model: https://huggingface.co/nvidia/parakeet-tdt_ctc-110m

