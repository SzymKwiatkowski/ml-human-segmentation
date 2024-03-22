import argparse
from pathlib import Path
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

from models.SegmentationModel import SegmentationModel
from utils.model_utils import export_onnx

class Exporter:
    @staticmethod
    def convert_32_to_16(onnx_model_fp32_path: str, onnx_model_fp16_path: str):
        onnx_model_fp32 = onnx.load(onnx_model_fp32_path)
        onnx_model_fp16 = float16.convert_float_to_float16(onnx_model_fp32)
        onnx.save(onnx_model_fp16, onnx_model_fp16_path)

    @staticmethod
    def convert_32_to_uint8(onnx_model_fp32_path: str, onnx_model_uint8_path: str):
        quant_pre_process(str(onnx_model_fp32_path), 'preprocessed_model_fp32.onnx')
        quantize_dynamic('preprocessed_model_fp32.onnx', onnx_model_uint8_path, weight_type=QuantType.QUInt8)

    @staticmethod
    def convert_to_onnx(to_convert_path: Path, save_converted_path: str = 'segmentation_model_fp32.onnx') -> str:
        model = SegmentationModel.load_from_checkpoint(to_convert_path)
        export_onnx(model, onnx_model_name=save_converted_path)

        return save_converted_path


def export(args):
    model_path = args.model_path
    exported_path = Exporter.convert_to_onnx(model_path)
    Exporter.convert_32_to_16(exported_path, 'segmentation_model_fp16.onnx')
    Exporter.convert_32_to_uint8(exported_path, 'segmentation_model_uint8.onnx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-m', '--model-path', action='store', default='best_model.ckpt')

    args_parsed = parser.parse_args()
    export(args_parsed)
