# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Summarizing abstract from covid19 publications."""

import json
import os

import pandas as pd

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@ONLINE {CORD-19-research-challenge,
    author = "An AI challenge with AI2, CZI, MSR, Georgetown, NIH & The White House",
    title  = "COVID-19 Open Research Dataset Challenge (CORD-19)",
    month  = "april",
    year   = "2020",
    url    = "https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge"
}
"""

_HOMEPAGE = "https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge"

_DESCRIPTION = """
CORD-19 is a resource of over 45,000 scholarly articles, including over 33,000
with full text, about COVID-19, SARS-CoV-2, and related coronaviruses.

To help organizing information in scientific literatures of COVID-19 through
abstractive summarization. This dataset parse those articles to pairs of
document and summaries of full_text-abstract or introduction-abstract.

Features includes strings of: abstract, introduction, full_text, sha, source_x,
title, doi, license, authors, publish_time, journal, url. The feature
"introduction" is created here by searching text segments indicating
introductions. Please refers to the covid-19 homepage for the rest features.
"""

_ABSTRACT = "abstract"
_INTRO_TEXT = "introduction"
_FULL_TEXT = "full_text"
_SHA = "sha"
_ADDITIONAL_FEATURES = [
    "sha", "source_x", "title", "doi", "license", "authors", "publish_time",
    "journal", "url"
]
_SEPARATOR = "\n"


class Covid19sumConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Covid19sum."""

  @tfds.core.disallow_positional_args
  def __init__(self, introduction_only=None, **kwargs):
    """BuilderConfig for Covid19sum.

    Args:
      introduction_only: wheter to use introduction as document or use full body
        text.
      **kwargs: keyword arguments forwarded to super.
    """
    # Version 1.0.0 uses 03/27/2020 data release.
    super(Covid19sumConfig, self).__init__(
        version=tfds.core.Version("1.0.0"), **kwargs)
    self.introduction_only = introduction_only


class Covid19sum(tfds.core.GeneratorBasedBuilder):
  """Covid19sum Dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
    This dataset need to be manually downloaded through kaggle api:
    `kaggle datasets download allen-institute-for-ai/CORD-19-research-challenge`
    Place the downloaded zip file in the manual folder
    (defaults to ~/tensorflow_datasets/manual/).
    """

  BUILDER_CONFIGS = [
      Covid19sumConfig(
          name="full_text",
          introduction_only=False,
          description="Use full text as document to generate abstract."),
      Covid19sumConfig(
          name="introduction",
          introduction_only=True,
          description="Use introduction as document to generate abstract,"
          "filter papers which do not have identifiable introduction secitons."
      ),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            k: tf.string for k in _ADDITIONAL_FEATURES +
            [_ABSTRACT, _FULL_TEXT, _INTRO_TEXT]
        }),
        supervised_keys=(_INTRO_TEXT if self.builder_config.introduction_only
                         else _FULL_TEXT, _ABSTRACT),
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    extracted_path = dl_manager.extract(
        os.path.join(dl_manager.manual_dir, "CORD-19-research-challenge.zip"))
    df = pd.read_csv(os.path.join(extracted_path, "metadata.csv")).fillna("")
    data_paths = []
    for _, row in df.iterrows():
      file_dir = row["full_text_file"]
      if row["has_full_text"] and _has_abstract(row) and file_dir:
        d = {k: row[k] for k in _ADDITIONAL_FEATURES + [_ABSTRACT, _SHA]}
        d["path"] = os.path.join(extracted_path, file_dir, file_dir,
                                 row[_SHA] + ".json")
        data_paths.append(d)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={"data_paths": data_paths},
        )
    ]

  def _generate_examples(self, data_paths=None):
    """Yields examples."""
    for d in data_paths:
      path = d.pop("path")
      if not tf.io.gfile.exists(path):
        continue
      with tf.io.gfile.GFile(path, "rb") as f:
        data_dict = json.load(f)
        body_text = data_dict.get("body_text", [])
        if not body_text:
          continue
        intro_text = _SEPARATOR.join(
            [seg["text"] for seg in body_text if _is_introduction(seg)])
        full_text = _SEPARATOR.join([seg["text"] for seg in body_text])
        if intro_text or not self.builder_config.introduction_only:
          d.update({_INTRO_TEXT: intro_text, _FULL_TEXT: full_text})
          yield d[_SHA], d


def _is_introduction(text_segment):
  return text_segment["section"].strip().lower().startswith("intro")


def _has_abstract(example):
  abstract = example[_ABSTRACT]
  return abstract and abstract.lower() != "unknown"
