# -*- coding: utf-8 -*-

import os
import time
from abc import ABCMeta
from typing import Optional, List, Any, Dict
import json
import configparser


import yaml
import requests


def build_recursion(obj: dict, prefix: str = "") -> Dict[str, Any]:
    results = {}

    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else key
        results[name] = value
        if isinstance(value, dict):
            results.update(build_recursion(value, name))

    return results


class AbstractConfig(metaclass=ABCMeta):
    """配置项抽象类"""

    def get(cls, name: str, default: Optional[str] = None, required: bool = True):
        """获取配置"""
        raise NotImplementedError(cls.get)


class EnvConfig(AbstractConfig):
    """环境变量配置项"""

    def get(self, name: str, default: Optional[str] = None, required: bool = True):
        """获取环境变量"""

        value = os.environ.get(name, default)
        if required and value is None:
            raise KeyError(f"Environemt variable {name} is not found")
        return value


class YamlConfig(AbstractConfig):
    def __init__(self, filepath: str = "./config.yml"):
        self.raw = None
        self.configs = None
        self.loaded = False

        self.load(filepath)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                self.raw = yaml.load(file, yaml.FullLoader)
            self.configs = build_recursion(self.raw)
        self.loaded = True

    def get(self, name: str, default: Optional[str] = None, required: bool = True):
        """获取yaml配置"""

        if not self.loaded:
            self.load()

        if self.configs is None:
            value = None
        else:
            value = self.configs.get(name, default)

        if required and value is None:
            raise KeyError(f"Yaml config {name} is not found")
        return value


class NacosConfig(AbstractConfig):
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        data_id: str,
        group: str,
        namespace: Optional[str] = None,
    ):
        self.host = host
        self.username = username
        self.password = password
        self.data_id = data_id
        self.group = group
        self.namespace = namespace

        self.access_token = None
        self.expired_at = None

        self.raw = None
        self.configs = None
        self.loaded = False

        self.load()

    @classmethod
    def from_config(cls, config: AbstractConfig):
        return cls(
            host=config.get("nacos.host"),
            username=config.get("nacos.username"),
            password=config.get("nacos.password"),
            data_id=config.get("nacos.data_id"),
            group=config.get("nacos.group"),
            namespace=config.get("nacos.namespace"),
        )

    def login(self, username: str, password: str):
        """登录Nacos"""

        resp = requests.post(
            f"{self.host}/nacos/v1/auth/login",
            data={"username": username, "password": password},
        )
        resp.raise_for_status()

        data = resp.json()
        self.access_token = data["accessToken"]
        self.expired_at = time.time() + data["tokenTtl"] - 1

    def read_properties(self, text: str):
        """读取Properties格式的配置"""

        config = configparser.ConfigParser()
        config.read_string("[DEFAULT]\n" + text)
        self.raw = dict(config["DEFAULT"])
        self.configs = build_recursion(self.raw)

    def read_json(self, text: str):
        """读取Json格式的配置"""
        self.raw = json.loads(text)
        self.configs = build_recursion(self.raw)

    def read_yaml(self, text: str):
        """读取Yaml格式的配置"""
        self.raw = yaml.load(text, yaml.FullLoader)
        self.configs = build_recursion(self.raw)

    def load(self):
        """加载Nacos配置"""

        if self.expired_at is None or time.time() > self.expired_at:
            self.login(self.username, self.password)

        resp = requests.get(
            f"{self.host}/nacos/v1/cs/configs",
            params={
                "dataId": self.data_id,
                "group": self.group,
                "tenant": self.namespace,
                "accessToken": self.access_token,
            },
        )
        resp.raise_for_status()

        config_type = resp.headers["Config-Type"]
        text = resp.text

        if config_type == "properties":
            self.read_properties(text)
        elif config_type == "json":
            self.read_json(text)
        elif config_type == "yaml":
            self.read_yaml(text)
        else:
            raise ValueError(f"Unsupported config type {config_type}")

        self.loaded = True

    def get(self, name: str, default: Optional[str] = None, required: bool = True):
        """获取Nacos配置"""

        if not self.loaded:
            self.load()

        if self.configs is None:
            value = None
        else:
            value = self.configs.get(name, default)

        if required and value is None:
            raise KeyError(f"Nacos config {name} is not found")
        return value


class Config(AbstractConfig):
    """配置项类"""

    adapters: List[AbstractConfig] = [
        EnvConfig(),
        YamlConfig(),
    ]
    prefix: str = ""

    @classmethod
    def set_prefix(cls, prefix: str):
        cls.prefix = prefix

    @classmethod
    def add_adapter(cls, adapter: AbstractConfig, index: Optional[int] = None):
        """添加配置适配器"""

        if index is not None:
            cls.adapters.insert(index, adapter)
        else:
            cls.adapters.append(adapter)

    @classmethod
    def get(cls, name: str, default: Optional[str] = None, required: bool = True):
        """自动获取配置"""

        if cls.prefix:
            name = cls.prefix + name
        values = [adapter.get(name, required=False) for adapter in cls.adapters]
        for value in values:
            if value is not None:
                return value

        if required:
            raise KeyError(f"{name} is not found")
        return default
