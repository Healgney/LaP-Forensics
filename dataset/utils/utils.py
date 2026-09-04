CAPTION_QUESTIONS = [
    'Could you please give me a detailed description of the image?',
    'Can you provide a thorough description of the this image?',
    'Please provide a thorough description of the this image',
    'Please provide a thorough description of the this image.',
    'Please describe in detail the contents of the image.',
    'Please describe in detail the contents of the image',
    'Could you give a comprehensive explanation of what can be found within this picture?',
    'Could you give me an elaborate explanation of this picture?',
    'Could you provide me with a detailed analysis of this photo?',
    'Could you please give me a detailed description of the image?',
    'Can you provide a thorough description of the this image?',
    'Please describe in detail the contents of the image',
    'Please describe in detail the contents of the image.',
    'Can you give a comprehensive explanation of this photo',
    'Please provide an elaborate explanation of this picture.',
    'Please provide an elaborate explanation of this picture',
    'Could you provide me with a detailed analysis of this photo',
]

REGION_QUESTIONS = [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
]

REGION_GROUP_QUESTIONS = [
    'Could you please give me a detailed description of these areas <region>?',
    'Can you provide a thorough description of the regions <region> in this image?',
    'Please describe in detail the contents of the boxed areas <region>.',
    'Could you give a comprehensive explanation of what can be found within <region> in the picture?',
    'Could you give me an elaborate explanation of the <region> regions in this picture?',
    'Can you provide a comprehensive description of the areas identified by <region> in this photo?',
    'Help me understand the specific locations labeled <region> in this picture in detail, please.',
    'What is the detailed information about the areas marked by <region> in this image?',
    'Could you provide me with a detailed analysis of the regions designated <region> in this photo?',
    'What are the specific features of the areas marked <region> in this picture that you can describe in detail?',
    'Could you elaborate on the regions identified by <region> in this image?',
    'What can you tell me about the areas labeled <region> in this picture?',
    'Can you provide a thorough analysis of the specific locations designated <region> in this photo?',
    'I am interested in learning more about the regions marked <region> in this image. Can you provide me with more information?',
    'Could you please provide a detailed description of the areas identified by <region> in this photo?',
    'What is the significance of the regions labeled <region> in this picture?',
    'I would like to know more about the specific locations designated <region> in this image. Can you provide me with more information?',
    'Can you provide a detailed breakdown of the regions marked <region> in this photo?',
    'What specific features can you tell me about the areas identified by <region> in this picture?',
    'Could you please provide a comprehensive explanation of the locations labeled <region> in this image?',
    'Can you provide a detailed account of the regions designated <region> in this photo?',
    'I am curious about the areas marked <region> in this picture. Can you provide me with a detailed analysis?',
    'What important details can you tell me about the specific locations identified by <region> in this image?',
    'Could you please provide a detailed description of the regions labeled <region> in this photo?',
    'What can you tell me about the features of the areas designated <region> in this picture?',
    'Can you provide a comprehensive overview of the regions marked <region> in this image?',
    'I would like to know more about the specific locations identified by <region> in this photo. Can you provide me with more information?',
    'What is the detailed information you have on the areas labeled <region> in this picture?',
    'Could you provide me with a thorough analysis of the regions designated <region> in this image?',
    'Can you provide a detailed explanation of the specific locations marked by <region> in this photo?'
]


SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

# GCG_QUESTIONS = [
#     'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
#     'Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
#     'Please describe in detail the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
#     'Could you give a comprehensive explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
#     'Could you give me an elaborate explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
#     'Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
# ]

GCG_QUESTIONS = [
    'Please provide a detailed analysis of artifacts in this photo, considering physical artifacts (e.g., optical display issues, violations of physical laws, and spatial/perspective errors), structural artifacts (e.g., deformed objects, asymmetry, or distorted text), and distortion artifacts (e.g., color/texture distortion, noise/blur, artistic style errors, and material misrepresentation). Output with interleaved segmentation masks for the corresponding parts of the answer.'
]


# =============================================================================
# Dual-Stream Architecture Prompts (for VAE reconstruction residual features)
# =============================================================================

DUAL_STREAM_GCG_QUESTIONS = [
    # Full version - Natural flow, correct localization logic
    """Analyze the provided dual-stream input (Original Image + Consistency Map) to detect synthetic artifacts. 

First, evaluate the Consistency Map. If it shows unnaturally low reconstruction error (smooth/dark regions), it strongly indicates synthetic generation. 
Then, scrutinize the Original Image to find specific evidence of generation flaws. Look for physical violations (e.g., incorrect lighting/shadows), structural anomalies (e.g., deformed anatomy, asymmetric objects), or textural distortions. 

Provide a coherent, natural analysis of your findings. When you identify specific, localized artifacts in the Original Image, output the interleaved segmentation masks ONLY for those anomalous local regions, not the entire low-residual area.""",

    # Concise version
    """Examine the Original Image and its Consistency Map. Smooth, dark regions in the map suggest synthetic origins due to low reconstruction error. 

Provide a detailed analysis of any physical, structural, or distortion artifacts found in the image. Output interleaved segmentation masks strictly for the specific deformed or anomalous objects identified, using the map as supporting evidence for their synthetic nature.""",

    # Guided version (If you need to force extreme focus)
    """Conduct a dual-stream forensic analysis. Use the Consistency Map (where darker areas = synthetic/low residual) as a global prior to confirm if the image is AI-generated. 

If synthetic patterns are confirmed, scan the Original Image to locate the exact source of failure (e.g., a fused limb, an impossible reflection, or corrupted text). Write a natural paragraph explaining these specific failures and output segmentation masks exclusively for the flawed components."""
]


DUAL_STREAM_SYSTEM_PROMPT = """You are LaP-Forensics, an expert system for detecting and explaining synthetic image artifacts. You process a dual-stream visual input:
1. **Original RGB Image**: For analyzing fine-grained visual details and semantics.
2. **Consistency Map (VAE Residuals)**: A global indicator of image origin. 

Key Physical Rule: AI-generated images typically exhibit LOWER reconstruction residuals (darker, smoother areas) because they align with the generative VAE distribution. Authentic natural images exhibit HIGHER residuals (brighter, noisy areas) due to complex high-frequency detail loss.

Your task is to synthesize information from both streams: use the Consistency Map to verify the overall synthetic nature of the content, and use the Original Image to pinpoint specific local anomalies."""

# =============================================================================
# Consistency Confirmation Templates for Answer Augmentation
# These templates are randomly inserted into answers during dual-stream training
# to teach the model to reference the residual map when explaining artifacts.
# =============================================================================

# Prefix templates: Added before the artifact phrase
CONSISTENCY_PREFIX_TEMPLATES = [
    "The consistency analysis reveals high reconstruction residuals in ",
    "Strong signals in the residual map indicate ",
    "The consistency map highlights abnormalities in ",
    "Elevated reconstruction errors are detected in ",
    "The residual analysis confirms synthetic patterns in ",
]

# Suffix templates: Added AFTER [SEG] token as post-annotation
# These should read naturally as sentence continuations
CONSISTENCY_SUFFIX_TEMPLATES = [
    " (confirmed by consistency analysis)",
    " (high residual deviation detected)",
    " (residual map shows synthetic pattern)",
    " (consistency check: elevated reconstruction error)",
    " (dual-stream verified)",
    ", as the consistency map confirms",
    ", which shows elevated residuals",
    ", verified by reconstruction analysis",
]

# Intro templates: Added at the beginning of the full answer
CONSISTENCY_INTRO_TEMPLATES = [
    "Based on both visual inspection and consistency analysis: ",
    "Combining visual and residual evidence: ",
    "The dual-stream analysis reveals: ",
    "Cross-referencing the image with the consistency map: ",
    "",  # Empty option for variety
]