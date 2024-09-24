import logging
import random
import warnings
import os
import gradio as gr
import numpy as np
import spaces
import torch
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from gradio_imageslider import ImageSlider
from PIL import Image
from huggingface_hub import snapshot_download

css = """
#col-container {
    margin: 0 auto;
    max-width: 512px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
    device = "cuda"
else:
    power_device = "CPU"
    device = "cpu"


huggingface_token = os.getenv("HUGGINFACE_TOKEN")

model_path = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev", 
    repo_type="model", 
    ignore_patterns=["*.md", "*..gitattributes"],
    local_dir="FLUX.1-dev",
    token=huggingface_token, # type a new token-id.
)


# Load pipeline
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=torch.bfloat16
).to(device)
pipe = FluxControlNetPipeline.from_pretrained(
    model_path, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to(device)

MAX_SEED = 1000000
MAX_PIXEL_BUDGET = 1024 * 1024


def process_input(input_image, upscale_factor, **kwargs):
    w, h = input_image.size
    w_original, h_original = w, h
    aspect_ratio = w / h

    was_resized = False

    if w * h * upscale_factor**2 > MAX_PIXEL_BUDGET:
        warnings.warn(
            f"Requested output image is too large ({w * upscale_factor}x{h * upscale_factor}). Resizing to ({int(aspect_ratio * MAX_PIXEL_BUDGET ** 0.5 // upscale_factor), int(MAX_PIXEL_BUDGET ** 0.5 // aspect_ratio // upscale_factor)}) pixels."
        )
        gr.Info(
            f"Requested output image is too large ({w * upscale_factor}x{h * upscale_factor}). Resizing input to ({int(aspect_ratio * MAX_PIXEL_BUDGET ** 0.5 // upscale_factor), int(MAX_PIXEL_BUDGET ** 0.5 // aspect_ratio // upscale_factor)}) pixels budget."
        )
        input_image = input_image.resize(
            (
                int(aspect_ratio * MAX_PIXEL_BUDGET**0.5 // upscale_factor),
                int(MAX_PIXEL_BUDGET**0.5 // aspect_ratio // upscale_factor),
            )
        )
        was_resized = True

    # resize to multiple of 8
    w, h = input_image.size
    w = w - w % 8
    h = h - h % 8

    return input_image.resize((w, h)), w_original, h_original, was_resized


@spaces.GPU(duration=42)
def infer(
    seed,
    randomize_seed,
    input_image,
    num_inference_steps,
    upscale_factor,
    controlnet_conditioning_scale,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    true_input_image = input_image
    input_image, w_original, h_original, was_resized = process_input(
        input_image, upscale_factor
    )

    # rescale with upscale factor
    w, h = input_image.size
    control_image = input_image.resize((w * upscale_factor, h * upscale_factor))

    generator = torch.Generator().manual_seed(seed)

    gr.Info("Upscaling image...")
    image = pipe(
        prompt="",
        control_image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        height=control_image.size[1],
        width=control_image.size[0],
        generator=generator,
    ).images[0]

    if was_resized:
        gr.Info(
            f"Resizing output image to targeted {w_original * upscale_factor}x{h_original * upscale_factor} size."
        )

    # resize to target desired size
    image = image.resize((w_original * upscale_factor, h_original * upscale_factor))
    image.save("output.jpg")
    # convert to numpy
    return [true_input_image, image, seed]


with gr.Blocks(css=css) as demo:
    # with gr.Column(elem_id="col-container"):
    gr.Markdown(
        f"""
    # ⚡ Flux.1-dev Upscaler ControlNet ⚡
    This is an interactive demo of [Flux.1-dev Upscaler ControlNet](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler) taking as input a low resolution image to generate a high resolution image.
    Currently running on {power_device}.

    *Note*: Even though the model can hamdle higher resolution images, due to GPU memory constraints, this demo was limited to a generated output not exceeding a pixel budget of 1024x1024. If the the requested size exceeds that limited, the input will be first resized keeping the aspect ratio such that the output of the controlNet model does not exceed the allocated pixel budget. The output is then resized to the targeted shape using a simple resizing. This may explain some artifacts for high resolution input. To adress this, run the demo locally or consider implementing a tiling strategy. Happy upscaling! 🚀
    """
    )

    with gr.Row():
        run_button = gr.Button(value="Run")

    with gr.Row():
        with gr.Column(scale=4):
            input_im = gr.Image(label="Input Image", type="pil")
        with gr.Column(scale=1):
            num_inference_steps = gr.Slider(
                label="Number of Inference Steps",
                minimum=8,
                maximum=50,
                step=1,
                value=28,
            )
            upscale_factor = gr.Slider(
                label="Upscale Factor",
                minimum=1,
                maximum=4,
                step=1,
                value=4,
            )
            controlnet_conditioning_scale = gr.Slider(
                label="Controlnet Conditioning Scale",
                minimum=0.1,
                maximum=1.5,
                step=0.1,
                value=0.6,
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=42,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

    with gr.Row():
        result = ImageSlider(label="Input / Output", type="pil", interactive=True)

    examples = gr.Examples(
        examples=[
            [42, False, "examples/image_1.jpg", 28, 4, 0.6],
            [42, False, "examples/image_2.jpg", 28, 4, 0.6],
            [42, False, "examples/image_3.jpg", 28, 4, 0.6],
            [42, False, "examples/image_4.jpg", 28, 4, 0.6],
            [42, False, "examples/image_5.jpg", 28, 4, 0.6],
            [42, False, "examples/image_6.jpg", 28, 4, 0.6],
        ],
        inputs=[
            seed,
            randomize_seed,
            input_im,
            num_inference_steps,
            upscale_factor,
            controlnet_conditioning_scale,
        ],
        fn=infer,
        outputs=result,
        cache_examples="lazy",
    )

    # examples = gr.Examples(
    #     examples=[
    #         #[42, False, "examples/image_1.jpg", 28, 4, 0.6],
    #         [42, False, "examples/image_2.jpg", 28, 4, 0.6],
    #         #[42, False, "examples/image_3.jpg", 28, 4, 0.6],
    #         #[42, False, "examples/image_4.jpg", 28, 4, 0.6],
    #         [42, False, "examples/image_5.jpg", 28, 4, 0.6],
    #         [42, False, "examples/image_6.jpg", 28, 4, 0.6],
    #         [42, False, "examples/image_7.jpg", 28, 4, 0.6],
    #     ],
    #     inputs=[
    #         seed,
    #         randomize_seed,
    #         input_im,
    #         num_inference_steps,
    #         upscale_factor,
    #         controlnet_conditioning_scale,
    #     ],
    # )

    gr.Markdown("**Disclaimer:**")
    gr.Markdown(
        "This demo is only for research purpose. Jasper cannot be held responsible for the generation of NSFW (Not Safe For Work) content through the use of this demo. Users are solely responsible for any content they create, and it is their obligation to ensure that it adheres to appropriate and ethical standards. Jasper provides the tools, but the responsibility for their use lies with the individual user."
    )
    gr.on(
        [run_button.click],
        fn=infer,
        inputs=[
            seed,
            randomize_seed,
            input_im,
            num_inference_steps,
            upscale_factor,
            controlnet_conditioning_scale,
        ],
        outputs=result,
        show_api=False,
        # show_progress="minimal",
    )

demo.queue().launch(share=False, show_api=False)
