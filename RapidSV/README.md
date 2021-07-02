# RapidSV: Accelerating Elliptic Curve Signature Verification

## Build

```
make volta  # For V100 GPU
make test   # Run test on sample data
make debug  # Enable debug logging
```

## Usage

```
./bin/gsv [GPU device ID=0]
```

## Acknowledgement
This project used the [CGBN](https://github.com/NVlabs/CGBN) library.
