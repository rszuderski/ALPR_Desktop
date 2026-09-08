#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lekkie modele widoku dla E2/Z2.

To warstwa pośrednia między domeną workflow a rendererami kampanii / free mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Step2CtaViewModel:
    label: str = ""
    command_id: str = ""
    command_context: dict = field(default_factory=dict)
    enabled: bool = True
    tone: str = "primary"


@dataclass(frozen=True)
class Step2ViewModel:
    stage_key: str = "step2"
    iteration_target: str = ""
    current_step: int = 0
    step2_status: str = "pending"
    state: str = "locked"
    title: str = "E2"
    summary: str = ""
    details: str = ""
    primary_cta: Step2CtaViewModel | None = None
    secondary_cta: Step2CtaViewModel | None = None
