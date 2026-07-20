import unittest

import torch

from SCTT_Net_manuscript0712_modified.gate.adaptive_fusion import AdaptiveFusion
from SCTT_Net_manuscript0712_modified.gate.gate_controller import GateController
from SCTT_Net_manuscript0712_modified.gate.temporal_memory import TemporalMemoryModule
from SCTT_Net_manuscript0712_modified.model.sctt_net import SCTTNet
from SCTT_Net_manuscript0712_modified.utils.metrics import (
    calculate_ioa,
    calculate_kge,
)


class TemporalMemoryTests(unittest.TestCase):
    def test_manuscript_windows(self):
        pcp = torch.arange(1.0, 9.0)
        sw = torch.full_like(pcp, 2.0)
        pet = torch.full_like(pcp, 3.0)
        x = torch.stack([pcp, sw, pet], dim=-1).unsqueeze(0)
        memory = TemporalMemoryModule(3, 7)(x)

        self.assertAlmostEqual(memory["pcp_3d"][0, -1].item(), 7.0)
        self.assertAlmostEqual(memory["pcp_7d"][0, -1].item(), 5.0)
        self.assertAlmostEqual(memory["sw_7d"][0, -1].item(), 2.0)
        self.assertAlmostEqual(memory["pet_7d"][0, -1].item(), 3.0)


class GateTests(unittest.TestCase):
    def test_zero_state_produces_half_gate(self):
        controller = GateController(dropout_rate=0.0, theta_init=0.0)
        for parameter in controller.basic_gate.parameters():
            torch.nn.init.zeros_(parameter)
        gate, components = controller(
            torch.zeros(2, 7, 3), return_components=True
        )
        self.assertTrue(torch.allclose(gate, torch.full_like(gate, 0.5)))
        self.assertTrue(
            torch.allclose(components["g_phy"], torch.full_like(gate, 0.5))
        )

    def test_fusion_direction_matches_equation_seven(self):
        fusion = AdaptiveFusion()
        global_features = torch.ones(1, 2, 3)
        local_features = torch.zeros(1, 2, 3)
        self.assertTrue(
            torch.equal(
                fusion(torch.ones(1, 2, 1), global_features, local_features),
                global_features,
            )
        )
        self.assertTrue(
            torch.equal(
                fusion(torch.zeros(1, 2, 1), global_features, local_features),
                local_features,
            )
        )


class ModelTests(unittest.TestCase):
    def test_multi_step_output_shape(self):
        for pred_len in (1, 3, 7):
            model = SCTTNet(
                num_meteo_features=5,
                pred_len=pred_len,
                seq_len=10,
                label_len=5,
                d_model=16,
                n_heads=4,
                e_layers=1,
                d_layers=1,
                d_ff=32,
                dropout=0.0,
                tcn_channels=[16, 16],
                tcn_dropout=0.0,
                gate_dropout=0.0,
            ).eval()
            meteo_input = {
                "enc_x": torch.randn(2, 10, 5),
                "dec_x": torch.randn(2, 5 + pred_len, 5),
            }
            gate_input = torch.randn(2, 10, 3)
            output, diagnostics = model(
                meteo_input, gate_input, return_diagnostics=True
            )
            self.assertEqual(tuple(output.shape), (2, pred_len, 1))
            self.assertEqual(tuple(diagnostics["gate"].shape), (2, pred_len, 1))


class MetricTests(unittest.TestCase):
    def test_perfect_kge_and_ioa(self):
        values = torch.tensor([1.0, 2.0, 4.0]).numpy()
        self.assertAlmostEqual(calculate_kge(values, values), 1.0)
        self.assertAlmostEqual(calculate_ioa(values, values), 1.0)


if __name__ == "__main__":
    unittest.main()
