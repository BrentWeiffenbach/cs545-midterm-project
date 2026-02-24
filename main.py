import argparse
import os
import subprocess
import sys


def run_inference(input_image, model_name, output_dir, use_fp16=False):
    inference_script = os.path.join(
        os.path.dirname(__file__),
        "src",
        "img2img-turbo",
        "src",
        "inference_unpaired.py",
    )
    cmd = [
        sys.executable,
        inference_script,
        "--model_name",
        model_name,
        "--input_image",
        input_image,
        "--output_dir",
        output_dir,
    ]
    if use_fp16:
        cmd.append("--use_fp16")

    print(f"Running img2img inference: {model_name}")
    result = subprocess.run(cmd, check=True)
    bname = os.path.basename(input_image)
    output_image = os.path.join(output_dir, bname)
    return output_image


def run_pipeline(day_image, night_image, results_folder):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from image_processing_pipeline import main as pipeline_main

    print("Running image processing pipeline...")
    pipeline_main(day_image, night_image, results_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Night-to-Day full pipeline")
    parser.add_argument(
        "--input_night_image",
        type=str,
        required=True,
        help="Path to the input night image",
    )
    parser.add_argument(
        "--input_day_image",
        type=str,
        default="data/day.jpg",
        help="Path to the ground truth day image",
    )
    parser.add_argument(
        "--inference_output_dir",
        type=str,
        default="outputs/inference",
        help="Directory for img2img output",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="Directory for final pipeline results",
    )
    parser.add_argument(
        "--use_fp16", action="store_true", help="Use FP16 for faster inference"
    )
    args = parser.parse_args()

    # Step 1: Run img2img-turbo inference
    img2img_output = run_inference(
        input_image=args.input_night_image,
        model_name="night_to_day",
        output_dir=args.inference_output_dir,
        use_fp16=args.use_fp16,
    )
    print(f"Inference output saved to: {img2img_output}")

    # Step 2: Run image processing pipeline on the inference output
    run_pipeline(
        day_image=args.input_day_image,
        night_image=img2img_output,
        results_folder=args.results_folder,
    )
    print(f"Final results saved to: {args.results_folder}")
