# Two-block 029 startup failure

The first launcher used an incorrect snapshot directory. The remote bootstrap script was absent, so the command exited 127 before model loading or training. The retry, `watching-two-block-029r`, validates the existing source directory before submission. This is an infrastructure failure, not a learning result.

See [failure.json](failure.json) for source provenance. Private infrastructure logs are excluded.
