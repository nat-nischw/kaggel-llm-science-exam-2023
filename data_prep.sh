#!/bin/bash

set -e # Stop script execution on any error
apt install git-lfs
git lfs install

mkdir -p ./asset/data/vector_index
mkdir -p ./asset/data/keyword_index/

clone_index() {
  local repo_url=$1
  local clone_path="./asset/data/$2"
  git clone "$repo_url" "$clone_path"
  find "$clone_path" -type f -name "*.index" -exec mv {} ./asset/data/vector_index/ \;
  rm -rf "$clone_path"
}


clone_index https://huggingface.co/datasets/natnitaract/teetouchjaknamon-faissbatchall-index-1 teetouchjaknamon-faissbatchall-index-1
clone_index https://huggingface.co/datasets/natnitaract/teetouchjaknamon-faissbatchall-index-2 teetouchjaknamon-faissbatchall-index-2

clone_keyword_index() {
  local repo_url=$1
  local clone_path="./$2"
  local move_to_path="./$3"
  
  git clone "$repo_url" "$clone_path"
  mv "$clone_path"/* "$move_to_path"
  rm -rf "$clone_path"
}


clone_keyword_index https://huggingface.co/datasets/natnitaract/wiki-scibatch-cohere-plaintext-banaei-9m wiki-scibatch-cohere-plaintext-banaei-9m ./asset/data/keyword_index/


echo "Dataset cloning and index file reorganization complete."

 