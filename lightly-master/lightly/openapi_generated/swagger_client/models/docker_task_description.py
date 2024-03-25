# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


from __future__ import annotations
import pprint
import re  # noqa: F401
import json


from typing import Union
from pydantic import Extra,  BaseModel, Field, StrictStr, confloat, conint
from lightly.openapi_generated.swagger_client.models.sampling_config import SamplingConfig
from lightly.openapi_generated.swagger_client.models.sampling_method import SamplingMethod

class DockerTaskDescription(BaseModel):
    """
    DockerTaskDescription
    """
    embeddings_filename: StrictStr = Field(..., alias="embeddingsFilename")
    embeddings_hash: StrictStr = Field(..., alias="embeddingsHash")
    method: SamplingMethod = Field(...)
    existing_selection_column_name: StrictStr = Field(..., alias="existingSelectionColumnName")
    active_learning_scores_column_name: StrictStr = Field(..., alias="activeLearningScoresColumnName")
    masked_out_column_name: StrictStr = Field(..., alias="maskedOutColumnName")
    sampling_config: SamplingConfig = Field(..., alias="samplingConfig")
    n_data: Union[confloat(ge=0, strict=True), conint(ge=0, strict=True)] = Field(..., alias="nData", description="the number of samples in the current embeddings file")
    __properties = ["embeddingsFilename", "embeddingsHash", "method", "existingSelectionColumnName", "activeLearningScoresColumnName", "maskedOutColumnName", "samplingConfig", "nData"]

    class Config:
        """Pydantic configuration"""
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        extra = Extra.forbid

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.dict(by_alias=by_alias))

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the model"""
        return json.dumps(self.to_dict(by_alias=by_alias))

    @classmethod
    def from_json(cls, json_str: str) -> DockerTaskDescription:
        """Create an instance of DockerTaskDescription from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of sampling_config
        if self.sampling_config:
            _dict['samplingConfig' if by_alias else 'sampling_config'] = self.sampling_config.to_dict(by_alias=by_alias)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerTaskDescription:
        """Create an instance of DockerTaskDescription from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DockerTaskDescription.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerTaskDescription) in the input: " + str(obj))

        _obj = DockerTaskDescription.parse_obj({
            "embeddings_filename": obj.get("embeddingsFilename"),
            "embeddings_hash": obj.get("embeddingsHash"),
            "method": obj.get("method"),
            "existing_selection_column_name": obj.get("existingSelectionColumnName"),
            "active_learning_scores_column_name": obj.get("activeLearningScoresColumnName"),
            "masked_out_column_name": obj.get("maskedOutColumnName"),
            "sampling_config": SamplingConfig.from_dict(obj.get("samplingConfig")) if obj.get("samplingConfig") is not None else None,
            "n_data": obj.get("nData")
        })
        return _obj

