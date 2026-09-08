#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared campaign data models used by wizard UI modules."""

from dataclasses import dataclass


@dataclass
class WizardStageStatus:
    key: str
    title: str
    state: str
    summary: str = ""
    details: str = ""
    primary_label: str = ""
    primary_command: object = None
    secondary_label: str = ""
    secondary_command: object = None
    badge_action_label: str = ""
    badge_action_command: object = None
    body_mode: str = ""
    body_visible: bool = False
    visible: bool = True
    is_current: bool = False
