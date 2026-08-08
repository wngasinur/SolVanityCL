import json
import logging
import multiprocessing as mp
import os
import platform
import secrets
import sys
import time
from datetime import datetime, timedelta
from math import ceil
from multiprocessing import Lock, Value

import pyopencl as cl

if os.environ.get("SOLVANITYCL_DEBUG_OPENCL"):
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
    os.environ["PYOPENCL_NO_CACHE"] = "TRUE"

from pathlib import Path

import click
import numpy as np
from base58 import b58decode, b58encode
from dotenv import load_dotenv
from nacl.signing import SigningKey

load_dotenv()

import threading

import runpod

from message import r

runpod.api_key = os.environ["RUNPOD_API_KEY"]

global_attempt = 0
current_hash_rate = 0.0
terminate_time = datetime.now()
topic = os.environ["TOPIC"]
# "O" means empty / not set (matches existing RunPod convention)
prefix = os.environ.get("PREFIX", "O")
suffix = os.environ.get("SUFFIX", "O")
case_sensitive = os.environ.get("CASE_SENSITIVE", "1")
terminate_seconds = int(os.environ["TERMINATE_SECONDS"])
iteration_bits_default = int(os.environ.get("ITERATION_BITS", "26"))
print("Version 1.0")
print(
    f"Subscriber topic {topic}, prefix={prefix}, suffix={suffix}, "
    f"case_sensitive={case_sensitive}, terminating in {terminate_seconds} seconds, "
    f"iteration_bits={iteration_bits_default}"
)
logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")


def terminate_function():
    global terminate_time
    terminate_time = datetime.now() + timedelta(seconds=terminate_seconds)
    time.sleep(terminate_seconds)
    logging.info("Thread %s: terminate_function timeout")
    publish(json.dumps({"state": "timeout"}), "timeout")
    runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])


x1 = threading.Thread(target=terminate_function, daemon=True)
x1.start()


def publish(message, state=None):
    try:
        # start/timeout/error/warning/progress/completed
        previous_state = r.get(
            f"runpod-{topic}-{prefix}-{suffix}-{case_sensitive}-stat"
        )
        if state != "completed" and previous_state == b"completed":
            raise RuntimeError("Already completed")
        elif state is not None:
            r.set(f"runpod-{topic}-{prefix}-{suffix}-{case_sensitive}-stat", state)
            if state == "completed":
                r.set(f"runpod-{topic}-last", message)

        r.publish(topic, message)
    except Exception as e:
        print(e)
        runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])


publish(json.dumps({"state": "start"}), "start")


class HostSetting:
    def __init__(self, kernel_source: str, iteration_bits: int) -> None:
        self.iteration_bits = iteration_bits
        self.iteration_bytes = np.ubyte(ceil(iteration_bits / 8))
        self.global_work_size = 1 << iteration_bits
        self.local_work_size = 32
        self.key32 = self.generate_key32()

        self.kernel_source = kernel_source

    def generate_key32(self):
        iteration_bytes = int(self.iteration_bytes)
        token_bytes = (
            secrets.token_bytes(32 - iteration_bytes) + b"\x00" * iteration_bytes
        )
        return np.frombuffer(token_bytes, dtype=np.ubyte).copy()

    def increase_key32(self):
        next_number = int.from_bytes(self.key32.tobytes(), "big") + (
            1 << self.iteration_bits
        )
        new_key32 = np.frombuffer(next_number.to_bytes(32, "big"), dtype=np.ubyte)
        carry_index = 32 - int(self.iteration_bytes)
        if (
            new_key32[carry_index] < self.key32[carry_index]
            and new_key32[carry_index] != 0
        ):
            new_key32[carry_index] = 0

        self.key32[:] = new_key32


def check_character(name: str, character: str):
    try:
        b58decode(character)
    except ValueError as e:
        logging.error(f"{str(e)} in {name}")
        publish(json.dumps({"state": "error", "message": str(e)}), "error")
        runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])
        sys.exit(1)
    except Exception as e:
        publish(json.dumps({"state": "warning", "message": str(e)}), "warning")
        raise e


def get_kernel_source(starts_with: str, ends_with: str, case_sensitive: int, cl):
    PREFIX_BYTES = list(bytes(starts_with.encode()))
    SUFFIX_BYTES = list(bytes(ends_with.encode()))

    with open(Path("opencl/kernel.cl"), "r") as f:
        source_lines = f.readlines()

    for i, s in enumerate(source_lines):
        if s.startswith("constant uchar PREFIX[]"):
            source_lines[i] = (
                f"constant uchar PREFIX[] = {{{', '.join(map(str, PREFIX_BYTES))}}};\n"
            )
        if s.startswith("constant uchar SUFFIX[]"):
            source_lines[i] = (
                f"constant uchar SUFFIX[] = {{{', '.join(map(str, SUFFIX_BYTES))}}};\n"
            )
        if s.startswith("constant int CASE_SENSITIVE"):
            source_lines[i] = f"constant int CASE_SENSITIVE = {{{case_sensitive}}};\n"

    source_str = "".join(source_lines)

    if cl.get_cl_header_version()[0] != 1 and platform.system() != "Windows":
        source_str = source_str.replace("#define __generic\n", "")

    return source_str


def get_all_gpu_devices():
    devices = []
    for ocl_platform in cl.get_platforms():
        try:
            devices.extend(ocl_platform.get_devices(device_type=cl.device_type.GPU))
        except cl.LogicError:
            logging.warning(
                "Skipping platform %s: no GPU devices available",
                ocl_platform.name,
            )
    if not devices:
        raise RuntimeError("No OpenCL GPU devices found")
    return [d.int_ptr for d in devices]


_worker_searcher = None
_worker_counter = Value("i", 0)
_worker_counter_lock = Lock()


def _init_gpu_worker(kernel_source: str, iteration_bits: int, gpu_count: int) -> None:
    global _worker_searcher
    with _worker_counter_lock:
        index = _worker_counter.value
        _worker_counter.value += 1

    if index >= gpu_count:
        raise RuntimeError(f"Pool worker index {index} exceeds GPU count {gpu_count}")

    setting = HostSetting(kernel_source, iteration_bits)
    _worker_searcher = Searcher(
        kernel_source=kernel_source,
        index=index,
        setting=setting,
        gpu_device_index=index,
        gpu_count=gpu_count,
    )
    logging.info(f"GPU worker {index} initialized")


def _gpu_search_batch(key32: np.ndarray) -> np.ndarray:
    global _worker_searcher
    try:
        return _worker_searcher.find(key32)
    except Exception as e:
        logging.exception(e)
        return np.zeros(33, dtype=np.ubyte)


def save_result(outputs, output_dir):
    result_count = 0
    for output in outputs:
        if not output[0]:
            continue
        result_count += 1
        pv_bytes = bytes(output[1:])
        pv = SigningKey(pv_bytes)
        pb_bytes = bytes(pv.verify_key)
        pubkey = b58encode(pb_bytes).decode()

        logging.info(f"Found: {pubkey}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        key = b58encode(bytes(list(pv_bytes + pb_bytes))).decode()
        json_keys = json.dumps({"key": key, "pubkey": pubkey})

        publish(json_keys, "completed")
        time.sleep(3)
        publish(json_keys, "completed")
        time.sleep(5)
        publish(json_keys, "completed")

        Path(output_dir, f"{pubkey}.json").write_text(key)
        runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])
    return result_count


class Searcher:
    def __init__(
        self,
        *,
        kernel_source,
        index: int,
        setting: HostSetting,
        context=None,
        device_int_ptr=None,
        gpu_device_index=None,
        gpu_count=None,
    ):
        if context:
            self.context = context
            self.gpu_chunks = 1
        else:
            if gpu_count is None:
                raise ValueError("gpu_count is required")
            if gpu_device_index is not None:
                device_ids = get_all_gpu_devices()
                if gpu_device_index >= len(device_ids):
                    raise RuntimeError(
                        f"GPU index {gpu_device_index} out of range "
                        f"(found {len(device_ids)} devices)"
                    )
                device = cl.Device.from_int_ptr(device_ids[gpu_device_index])
            elif device_int_ptr is not None:
                device = cl.Device.from_int_ptr(device_int_ptr)
            else:
                raise ValueError("gpu_device_index or device_int_ptr is required")
            self.context = cl.Context([device])
            self.gpu_chunks = gpu_count
        self.command_queue = cl.CommandQueue(self.context)

        self.setting = setting
        self.index = index

        program = cl.Program(self.context, kernel_source).build()
        self.kernel = cl.Kernel(program, "generate_pubkey")

        self.memobj_key32 = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, 32)
        self.memobj_output = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, 33)
        self.memobj_occupied_bytes = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.array([self.setting.iteration_bytes]),
        )
        self.memobj_group_offset = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.array([self.index]),
        )
        self._output = np.zeros(33, dtype=np.ubyte)
        self._zero_output = np.zeros(33, dtype=np.ubyte)

        self.kernel.set_arg(0, self.memobj_key32)
        self.kernel.set_arg(1, self.memobj_output)
        self.kernel.set_arg(2, self.memobj_occupied_bytes)
        self.kernel.set_arg(3, self.memobj_group_offset)

    def find(self, key32=None):
        if key32 is None:
            key32 = self.setting.key32

        cl.enqueue_copy(self.command_queue, self.memobj_key32, key32)
        cl.enqueue_copy(self.command_queue, self.memobj_output, self._zero_output)

        global_worker_size = self.setting.global_work_size // self.gpu_chunks
        cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_worker_size,),
            (self.setting.local_work_size,),
        )
        cl.enqueue_copy(self.command_queue, self._output, self.memobj_output).wait()

        return self._output.copy()


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option(
    "--starts-with",
    type=str,
    help="Public key starts with the indicated prefix.",
    default="",
)
@click.option(
    "--ends-with",
    type=str,
    help="Public key ends with the indicated suffix.",
    default="",
)
@click.option(
    "--case-sensitive",
    type=int,
    help="",
    default=1,
)
@click.option(
    "--count",
    type=int,
    help="Count of pubkeys to generate.",
    default=1,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Output directory.",
    default="./",
)
@click.option(
    "--select-device/--no-select-device",
    type=bool,
    help="Select OpenCL device manually",
    default=False,
)
@click.option(
    "--iteration-bits",
    type=int,
    help=(
        "Number of iteration occupied bits. Recommended 24, 26, 28, 30, 32. "
        "Larger bits = longer batch. Defaults to ITERATION_BITS env var, else 26."
    ),
    default=None,
)
@click.pass_context
def search_pubkey(
    ctx,
    starts_with: str,
    ends_with: str,
    case_sensitive: int,
    count: int,
    output_dir: str,
    select_device: bool,
    iteration_bits: int,
):
    """Search Solana vanity pubkey"""
    global global_attempt

    if iteration_bits is None:
        iteration_bits = iteration_bits_default
    if iteration_bits < 16 or iteration_bits > 32:
        logging.error(
            "iteration-bits must be between 16 and 32 (got %s)", iteration_bits
        )
        publish(
            json.dumps(
                {
                    "state": "error",
                    "message": f"invalid iteration-bits: {iteration_bits}",
                }
            ),
            "error",
        )
        runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])
        sys.exit(1)

    if not starts_with and not ends_with:
        print("Please provides at least [starts with] or [ends with]\n")
        click.echo(ctx.get_help())
        sys.exit(1)

    check_character("starts_with", starts_with)
    check_character("ends_with", ends_with)

    logging.info(
        f"Searching Solana pubkey that starts with '{starts_with}' and ends with '{ends_with}' "
        f"(iteration_bits={iteration_bits}, batch={1 << iteration_bits})"
    )
    gpu_counts = len(get_all_gpu_devices())
    logging.info(f"Found {gpu_counts} OpenCL GPU(s) in main process")

    kernel_source = get_kernel_source(starts_with, ends_with, case_sensitive, cl)
    setting = HostSetting(kernel_source, iteration_bits)
    result_count = 0

    logging.info(f"Searching with {gpu_counts} OpenCL devices")
    publish(json.dumps({"state": "progress"}), "progress")

    def heartbeat_function():
        global global_attempt, current_hash_rate, terminate_time, terminate_seconds
        while True:
            y = terminate_time - datetime.now()
            elapsedTime = terminate_seconds - y.seconds
            publish(
                json.dumps(
                    {
                        "attempt": str(round(global_attempt, 2)),
                        "elapsedTime": elapsedTime,
                        "maxTime": terminate_seconds,
                        "hashRate": round(current_hash_rate, 2),
                    }
                )
            )
            chars = len("" if prefix == "O" else prefix) + len(
                "" if suffix == "O" else suffix
            )
            if chars >= 6:
                time.sleep(30)
            elif chars >= 5:
                time.sleep(20)
            else:
                time.sleep(7)

    x2 = threading.Thread(target=heartbeat_function, daemon=True)
    x2.start()

    def run_batch(batch_fn):
        global global_attempt, current_hash_rate
        st = time.time()
        output = batch_fn()
        elapsed = time.time() - st
        attempts = setting.global_work_size / 1e6
        if elapsed > 0:
            current_hash_rate = attempts / elapsed
        global_attempt += attempts
        return output

    if select_device:
        context = cl.create_some_context()
        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=0,
            setting=setting,
            context=context,
        )
        while result_count < count:
            output = run_batch(lambda: searcher.find(setting.key32))
            setting.increase_key32()
            result_count += save_result([output], output_dir)
        return

    if gpu_counts == 1:
        logging.info("Using single-GPU in-process search (no multiprocessing)")
        searcher = Searcher(
            kernel_source=kernel_source,
            index=0,
            setting=setting,
            gpu_device_index=0,
            gpu_count=1,
        )
        while result_count < count:
            output = run_batch(lambda: searcher.find(setting.key32))
            setting.increase_key32()
            result_count += save_result([output], output_dir)
        return

    logging.info("Using spawn-based multiprocessing for %s GPUs", gpu_counts)
    with _worker_counter_lock:
        _worker_counter.value = 0

    mp_context = mp.get_context("spawn")
    with mp_context.Pool(
        processes=gpu_counts,
        initializer=_init_gpu_worker,
        initargs=(kernel_source, iteration_bits, gpu_counts),
    ) as pool:
        while result_count < count:
            key32_snapshot = setting.key32.copy()
            results = run_batch(
                lambda: pool.map(_gpu_search_batch, [key32_snapshot] * gpu_counts)
            )
            result_count += save_result(results, output_dir)
            setting.increase_key32()


@cli.command(context_settings={"show_default": True})
def show_device():
    """Show OpenCL devices"""

    platforms = cl.get_platforms()

    for p_index, ocl_platform in enumerate(platforms):
        print(f"Platform {p_index}: {ocl_platform.name}")

        try:
            devices = ocl_platform.get_devices()
        except cl.LogicError as e:
            print(f"- Failed to enumerate devices: {e}")
            continue

        for d_index, device in enumerate(devices):
            print(f"- Device {d_index}: {device.name}")


if __name__ == "__main__":
    cli()
