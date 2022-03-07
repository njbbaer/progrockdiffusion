# What do all the settings do?
This document hopes to explain what the various settings are. Some of them we're still figuring out. :)

| *Setting name* | *Default in settings.json* | *Explanation*
| -------------------------|------------------------|:---
| **batch_name** | "Default" | The directory within images_out to store your results
| **text_prompts** | "The Big Sur Coast, by Asher Brown Durand, featured on ArtStation." | The phrase(s) to use for generating an image
| **n_batches** | 1 | How many images to generate
| **steps** | 500 | Generally, the more steps you run, the more detailed the results. At least 250 is recommended.
| **display_rate** | 50 | How often (in steps) to update the progress.png image
| **width** | 768 | Image output width in pixels - must be a multiple of 64
| **height** | 448 | Image output height in pixels - must be a multiple of 64
| **set_seed** | "random_seed" | If set to random_seed it will generate a new seed. Replace this with a specific number to elimate randomness in the start
| **image_prompts** | {} | For using images instead of words for prompts. Not recommended.
| **clip_guidance_scale** | 5000 | Controls how much the image should look like the prompt.
| **tv_scale** | 0 | Controls the smoothness of the final output.
| **range_scale** | 150 | Controls how far out of range RGB values are allowed to be.
| **sat_scale** | 0 | Controls how much saturation is allowed.
| **cutn_batches** | 4 | Lowering this number can reduce how much memory is needed, however note that cutn itself is hard set at 16
| **max_frames** | 10000 | No idea
| **interp_spline** | "Linear" | Do not change, currently will not look good.
| **init_image** | null | The starting image to use. Usuallly leave this blank and it will start with randomness
| **init_scale** | 1000 | This enhances the effect of the init image, a good value is 1000
| **skip_steps** | 0 | How many steps in the overall process to skip. Generally leave this at 0, though if using an init_image it is recommended to be 50% of overall steps
| **frames_scale** | 1500 | Tries to guide the new frame to looking like the old one. A good default is 1500.
| **frames_skip_steps** | "60%" | Will blur the previous frame - higher values will flicker less
| **perlin_init** | false | Option to start with random perlin noise
| **perlin_mode** | "mixed" | Other options are "grey" or "color", what they do I'm not sure
| **skip_augs** | false | Controls whether to skip torchvision augmentations
| **randomize_class** | true | Controls whether the imagenet class is randomly changed each iteration
| **clip_denoised** | false | Determines whether CLIP discriminates a noisy or denoised image
| **clamp_grad** | true | Experimental: Using adaptive clip grad in the cond_fn
| **clamp_max** | 0.05 | Anyone? Beuller?
| **fuzzy_prompt** | false | Controls whether to add multiple noisy prompts to the prompt losses
| **rand_mag** | 0.05 | Controls the magnitude of the random noise
| **eta** | 0.8 | Has to do with how much the generator can stray from your prompt, apparently.
| **diffusion_model** | "512x512_diffusion_uncond_finetune_008100",
| **use_secondary_model** | true | Reduces memory and improves speed, potentially at a loss of quality
| **diffusion_steps** | 1000 | Note: The code seems to calculate this no matter what you put in, so might as well leave it
| **ViTB32** | false | Enable or disable the VitB32 CLIP model. Low memory, low accuracy
| **ViTB16** | true | Enable or disable the VitB16 CLIP model. Med memory, high accuracy
| **ViTL14** | false | Enable or disable the VitB32 CLIP model. Very high memory, very high accuracy
| **RN101** | false | Enable or disable the VitB32 CLIP model. Low memory, low accuracy
| **RN50** | true | Enable or disable the VitB32 CLIP model. Med memory, med accuracy
| **RN50x4** | false | Enable or disable the VitB32 CLIP model. High memory, high accuracy
| **RN50x16** | false | Enable or disable the VitB32 CLIP model. Very high memory, high accuracy
| **SLIPB16** | false | Enable or disable the SLIPB16 CLIP model. High memory, weird accuracy
| **SLIPL16** | false | Enable or disable the SLIPL16 CLIP model. Very high memory, unknown accuracy
| **cut_overview** | "[12]\*400+[4]\*600" | How many "big picture" passes to do. More towards the start, less later, is the general idea
| **cut_innercut** | "[4]\*400+[12]\*600" | Conversely, how many detail passes to do. Fewer at the start, then get more detailed
| **cut_ic_pow** | 1 | Anyone? Beuller?
| **cut_icgray_p** | "[0.2]\*400+[0]\*600" | Anyone? Beuller?
| **key_frames** | true | Animation stuff...
| **angle** | "0:(0)"| Animation stuff...
| **zoom** | "0: (1), 10: (1.05)" | Animation stuff...
| **translation_x** | "0: (0)" | Animation stuff...
| **translation_y** | "0: (0)" | Animation stuff...
| **video_init_path** | "/content/training.mp4"| Animation stuff...
| **extract_nth_frame** | 2 | Animation stuff...
