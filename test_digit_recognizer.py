import contextlib
import io
import os
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


_STUBBED_MODULE_NAMES = (
    "tkinter",
    "tkinter.ttk",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
)
_MISSING_MODULE = object()
_ORIGINAL_MODULES = {
    name: sys.modules.get(name, _MISSING_MODULE)
    for name in _STUBBED_MODULE_NAMES
}

tkinter = types.ModuleType("tkinter")
ttk = types.ModuleType("tkinter.ttk")
tkinter.ttk = ttk
sys.modules["tkinter"] = tkinter
sys.modules["tkinter.ttk"] = ttk

pil = types.ModuleType("PIL")
pil_image = types.ModuleType("PIL.Image")
pil_image_draw = types.ModuleType("PIL.ImageDraw")
pil.Image = pil_image
pil.ImageDraw = pil_image_draw
sys.modules["PIL"] = pil
sys.modules["PIL.Image"] = pil_image
sys.modules["PIL.ImageDraw"] = pil_image_draw

from digit_recognizer import (  # noqa: E402
    EXPECTED_NUM_INPUTS,
    EXPECTED_NUM_OUTPUTS,
    MAX_NUM_HIDDEN,
    MODEL_FLOAT64_FORMAT,
    MODEL_HEADER_FORMAT,
    MNISTModel,
    main,
)


def tearDownModule():
    for name, original in _ORIGINAL_MODULES.items():
        if original is _MISSING_MODULE:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class MNISTModelLoadTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_model_file(self, data):
        path = os.path.join(self.temp_dir.name, "model.bin")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def pack_header(
        self,
        num_inputs=EXPECTED_NUM_INPUTS,
        num_hidden=2,
        num_outputs=EXPECTED_NUM_OUTPUTS,
    ):
        return struct.pack(MODEL_HEADER_FORMAT, num_inputs, num_hidden, num_outputs)

    def pack_payload(self, num_hidden=2):
        value_count = (
            EXPECTED_NUM_INPUTS * num_hidden
            + num_hidden
            + num_hidden * EXPECTED_NUM_OUTPUTS
            + EXPECTED_NUM_OUTPUTS
        )
        return struct.pack(f"<{value_count}d", *([0.0] * value_count))

    def load_model(self, data):
        path = self.write_model_file(data)
        with contextlib.redirect_stdout(io.StringIO()):
            return MNISTModel().load_model(path)

    def assert_load_raises(self, data, expected_message):
        with self.assertRaises(ValueError) as context:
            self.load_model(data)
        self.assertIn(expected_message, str(context.exception))

    def test_truncated_header_raises_descriptive_error(self):
        self.assert_load_raises(b"\x00" * 8, "header incomplete")

    def test_negative_num_inputs_raises_descriptive_error(self):
        self.assert_load_raises(
            self.pack_header(num_inputs=-1),
            "num_inputs",
        )

    def test_negative_num_hidden_raises_descriptive_error(self):
        self.assert_load_raises(
            self.pack_header(num_hidden=-1),
            "num_hidden",
        )

    def test_negative_num_outputs_raises_descriptive_error(self):
        self.assert_load_raises(
            self.pack_header(num_outputs=-1),
            "num_outputs",
        )

    def test_zero_num_inputs_raises_descriptive_error(self):
        self.assert_load_raises(
            self.pack_header(num_inputs=0),
            "num_inputs",
        )

    def test_zero_num_hidden_raises_descriptive_error(self):
        self.assert_load_raises(
            self.pack_header(num_hidden=0),
            "num_hidden",
        )

    def test_zero_num_outputs_raises_descriptive_error(self):
        self.assert_load_raises(
            self.pack_header(num_outputs=0),
            "num_outputs",
        )

    def test_wrong_num_inputs_raises_architecture_mismatch(self):
        self.assert_load_raises(
            self.pack_header(num_inputs=EXPECTED_NUM_INPUTS + 1),
            "architecture mismatch",
        )

    def test_wrong_num_outputs_raises_architecture_mismatch(self):
        self.assert_load_raises(
            self.pack_header(num_outputs=EXPECTED_NUM_OUTPUTS + 1),
            "architecture mismatch",
        )

    def test_oversized_num_hidden_raises_descriptive_error(self):
        self.assert_load_raises(
            self.pack_header(num_hidden=MAX_NUM_HIDDEN + 1),
            "num_hidden",
        )

    def test_truncated_weight_section_names_affected_section(self):
        self.assert_load_raises(
            self.pack_header(num_hidden=2) + struct.pack(MODEL_FLOAT64_FORMAT, 0.0),
            "hidden weights",
        )

    def test_valid_model_loads_successfully(self):
        path = self.write_model_file(
            self.pack_header(num_hidden=2) + self.pack_payload(2)
        )
        model = MNISTModel()

        with contextlib.redirect_stdout(io.StringIO()):
            model.load_model(path)

        self.assertEqual(model.hidden_weights.shape, (EXPECTED_NUM_INPUTS, 2))
        self.assertEqual(model.hidden_biases.shape, (2,))
        self.assertEqual(model.output_weights.shape, (2, EXPECTED_NUM_OUTPUTS))
        self.assertEqual(model.output_biases.shape, (EXPECTED_NUM_OUTPUTS,))

    def test_main_exits_cleanly_for_corrupted_model(self):
        with mock.patch.object(sys, "argv", ["digit_recognizer.py", "serial"]):
            with mock.patch.object(
                MNISTModel,
                "load_model",
                side_effect=ValueError("bad header"),
            ):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    with self.assertRaises(SystemExit) as context:
                        main()

        self.assertEqual(context.exception.code, 1)
        self.assertIn("Error loading model: bad header", output.getvalue())


if __name__ == "__main__":
    unittest.main()
