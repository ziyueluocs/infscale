import torch
import deepspeed
from transformers import BertForSequenceClassification, BertTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile, FlopsProfiler
from deepspeed.profiling.flops_profiler import FlopsProfiler, num_to_string, number_to_string, macs_to_string, params_to_string, duration_to_string, flops_to_string
from deepspeed.accelerator import get_accelerator
from profile_in_json import json_dump_model_profile, json_dump_model_aggregated_profile

def bert_input_constructor(batch_size, seq_len, tokenizer, batch_num=1):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * (batch_size * batch_num),
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * (batch_size * batch_num))
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs

def get_model_training_profile(model,
                      input_shape=None,
                      args=[],
                      kwargs={},
                      print_profile=True,
                      detailed=True,
                      module_depth=-1,
                      top_modules=1,
                      warm_up=1,
                      as_string=True,
                      output_file=None,
                      labels=None,
                      ignore_modules=None):
    """Returns the total floating-point operations, MACs, and parameters of a model.
    Example:
    .. code-block:: python
        model = torchvision.models.alexnet()
        batch_size = 256
        flops, macs, params = get_model_profile(model=model, input_shape=(batch_size, 3, 224, 224)))
    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        input_shape (tuple): input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args (list): list of positional arguments to the model.
        kwargs (dict): dictionary of keyword arguments to the model.
        print_profile (bool, optional): whether to print the model profile. Defaults to True.
        detailed (bool, optional): whether to print the detailed model profile. Defaults to True.
        module_depth (int, optional): the depth into the nested modules. Defaults to -1 (the inner most modules).
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        warm_up (int, optional): the number of warm-up steps before measuring the latency of each module. Defaults to 1.
        as_string (bool, optional): whether to print the output as string. Defaults to True.
        output_file (str, optional): path to the output file. If None, the profiler prints to stdout.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.
    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    assert isinstance(model, torch.nn.Module), "model must be a PyTorch module"
    print(get_accelerator().communication_backend_name())
    ds_engine = deepspeed.initialize(model=model, config="ds_config.json")
    prof = FlopsProfiler(model, ds_engine=ds_engine)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()

    if input_shape is not None:
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"
        try:
            # create fake input
            input = torch.ones(()).new_empty(
                (*input_shape, ),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
        except StopIteration:
            input = torch.ones(()).new_empty((*input_shape, ))

        args = [input]
        # create fake label
        labels = torch.ones(len(input))
    
    if "labels" not in kwargs:
        assert labels != None
        kwargs["labels"] = labels

    assert (len(args) > 0) or (len(kwargs) > 0), "args and/or kwargs must be specified if input_shape is None"
    # warm up steps
    for _ in range(warm_up):
        if kwargs:
            outputs = model(*args, **kwargs)
        else:
            outputs = model(*args, labels=labels)
        loss = outputs[0]
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    prof.start_profile(ignore_list=ignore_modules)
    # profiling steps
    if kwargs:
        outputs = model(*args, **kwargs)
    else:
        outputs= model(*args, labels=labels)
    optimizer.zero_grad()

    loss = outputs[0]
    loss.backward()
    optimizer.step()

    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        json_dump_model_profile(model, prof, profile_step=warm_up,
                                 module_depth=module_depth,
                                 top_modules=top_modules,
                                 detailed=detailed,
                                 output_file=output_file)

    prof.end_profile()
    if as_string:
        return number_to_string(flops), macs_to_string(macs), params_to_string(params)

    return flops, macs, params

with get_accelerator().device(0):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    batch_size = 4
    seq_len = 128
    enable_profile = True
    dataset = bert_input_constructor(batch_size, seq_len, tokenizer, batch_num=100)
    flops, macs, params = get_model_training_profile(
        model,
        kwargs=dataset,
        print_profile=True,
        labels=dataset["labels"],
        detailed=True,
        output_file="./bert_training_profile.json"
    )
