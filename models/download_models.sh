#!/usr/bin/env bash
set -eux

download() {
  wget "$1" -O "$2"
}

download "https://www.dropbox.com/s/4i99mzddk95edyq/tsn_rgb.ckpt?dl=1" tsn_rgb.ckpt
download "https://www.dropbox.com/s/res0i1ns7v30g9y/tsn_flow.ckpt?dl=1" tsn_flow.ckpt
download "https://www.dropbox.com/s/l1cs7kozz3f03r4/trn_rgb.ckpt?dl=1" trn_rgb.ckpt
download "https://www.dropbox.com/s/4rehj36vyip82mu/trn_flow.ckpt?dl=1" trn_flow.ckpt
download "https://www.dropbox.com/s/5yxnzubch7b6niu/tsm_rgb.ckpt?dl=1" tsm_rgb.ckpt
download "https://www.dropbox.com/s/8x9hh404k641rqj/tsm_flow.ckpt?dl=1" tsm_flow.ckpt
