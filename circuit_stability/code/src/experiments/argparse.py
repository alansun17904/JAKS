from argparse import Namespace

def cs_entrypoint(
    model="gpt2",
    batch_size=1,
    ndevices=1,
    device="cuda",
    seed=42,
    dataset="custom",
    format="zero-shot",
    extraction="tail",
    ig_steps = 5,
    data_params = None,
    format_params = None,
    patching_metric = "kl"
):
    # lazy import avoids heavy imports & circulars at package import time
    from experiments.circuit_discovery import main as _cs_main

    opts = Namespace(
        model_name=model,
        batch_size=batch_size,
        ndevices=ndevices,
        device=device,
        seed=seed,
        dataset=dataset,
        format=format,
        extraction=extraction,
        ig_steps=ig_steps,
        data_params=data_params,
        format_params=format_params,
        patching_metric=patching_metric
    )
    return _cs_main(opts)
