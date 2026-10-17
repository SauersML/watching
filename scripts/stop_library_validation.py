"""Run-specific independent stop timer; never prints or writes authentication data."""

import json
import os
import sys
import time
import urllib.request


def credentials(expected):
    environment = dict(os.environ)
    for item in open("/proc/1/environ", "rb").read().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            if key in (b"RUNPOD_API_KEY", b"RUNPOD_POD_ID"):
                environment.setdefault(key.decode(), value.decode())
    if environment.get("RUNPOD_POD_ID") != expected or not environment.get("RUNPOD_API_KEY"):
        raise RuntimeError("Run-specific autonomous stop credentials unavailable")
    return environment["RUNPOD_API_KEY"]


def main():
    mode, pod = sys.argv[1:3]
    key = credentials(pod)
    if mode == "preflight":
        print(json.dumps({"autonomous_stop_preflight": True, "pod": pod}))
        return
    if mode == "watchdog":
        time.sleep(max(0, int(sys.argv[3]) - 20))
    elif mode != "stop":
        raise ValueError("Unknown stop mode")
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                "https://rest.runpod.io/v1/pods/" + pod + "/stop",
                data=b"",
                method="POST",
                headers={"Authorization": "Bearer " + key},
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                print(json.dumps({"stop_requested": True, "pod": pod, "status": response.status}))
            return
        except Exception as error:
            print(
                json.dumps({"stop_attempt": attempt + 1, "error_type": type(error).__name__}),
                flush=True,
            )
            time.sleep(2)
    raise RuntimeError("Run-specific stop could not be confirmed")


if __name__ == "__main__":
    main()
