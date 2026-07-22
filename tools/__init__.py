"""Runtime package marker for explicitly shipped research dependencies.

Qu-v2 production weights are content-locked to
``tools.research.qu_v2a_features``.  Shipping this marker makes the local
package win over any unrelated third-party ``tools`` package installed in the
competition image; without it, Python may discard our namespace package and
the policy silently falls back to rules.
"""
