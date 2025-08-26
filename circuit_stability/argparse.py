from argparse import Namespace

def cs_entrypoint(
    model="gpt2",
    ofile=None,
    batch_size=1,
    ndevices=1,
    device="cuda",
    seed=42,
    dataset="custom",
    format="zero-shot",
    extraction="last_token",
):
    # lazy import avoids heavy imports & circulars at package import time
    from code.src.experiments.circuit_discovery import main as _cs_main

    opts = Namespace(
        model_name=model,
        ofname=ofile,
        batch_size=batch_size,
        ndevices=ndevices,
        device=device,
        seed=seed,
        dataset=dataset,
        format=format,
        extraction=extraction,
    )
    return _cs_main(opts)
