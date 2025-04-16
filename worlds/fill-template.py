#!/usr/bin/env python
"""
First Approach:
A helper script that reads a Webots world template file and replaces the
'%PLACEHOLDER%' with brick definitions arranged in a grid.

Usage example:
    python fill-template.py --nx 4 --ny 4 --nz 5 --tx -5.7 --ty -0.36 --tz 0.3465 --template template.wbt --output output.wbt
"""

import argparse
from typing import Tuple

BRICK_SIZE_X: float = 0.247
BRICK_SIZE_Y: float = 0.3
BRICK_SIZE_Z: float = 0.25


def generate_brick_string(translation: Tuple[float, float, float]) -> str:
    """
    Generate a Solid node string for a brick at the specified translation.

    Args:
        translation: A tuple of (x, y, z) coordinates for the brick translation.

    Returns:
        A string representing the brick's Solid node definition.
    """
    x, y, z = translation
    brick_str: str = (
        f"Solid {{\n"
        f"  translation {x:.6f} {y:.6f} {z:.6f}\n"
        f"  rotation 0 0 1 1.5708\n"
        f"  children [\n"
        f"    TexturedBoxShape {{\n"
        f"      size {BRICK_SIZE_X} {BRICK_SIZE_Y} {BRICK_SIZE_Z}\n"
        f"    }}\n"
        f"  ]\n"
        f"  boundingObject Box {{\n"
        f"    size {BRICK_SIZE_X} {BRICK_SIZE_Y} {BRICK_SIZE_Z}\n"
        f"  }}\n"
        f"  physics Physics {{\n"
        f"  }}\n"
        f"}}\n"
    )
    return brick_str


def generate_bricks_placeholder(
    nx: int, ny: int, nz: int, first_translation: Tuple[float, float, float]
) -> str:
    """
    Generate the placeholder string content for bricks arranged in a grid.

    Args:
        nx: Number of bricks in the x direction.
        ny: Number of bricks in the y direction.
        nz: Number of bricks in the z direction.
        first_translation: Translation (x, y, z) for the first brick.

    Returns:
        A string with all brick nodes concatenated.
    """
    tx, ty, tz = first_translation
    bricks_str: str = ""
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x: float = tx + i * BRICK_SIZE_Y
                y: float = ty + j * BRICK_SIZE_X
                z: float = tz + k * BRICK_SIZE_Z
                bricks_str += generate_brick_string((x, y, z))
    return bricks_str


def process_template(template_path: str, output_path: str, bricks_placeholder: str) -> None:
    """
    Process the template file and replace the placeholder with the provided brick definitions.

    Args:
        template_path: Path to the template world file.
        output_path: Path to save the updated world file.
        bricks_placeholder: The brick definitions string to replace the placeholder.
    """
    with open(template_path, "r") as f:
        content: str = f.read()
    content = content.replace("%PLACEHOLDER%", bricks_placeholder)
    with open(output_path, "w") as f:
        f.write(content)


def main() -> None:
    """
    Main function to generate a Webots world file with bricks arranged in a grid.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate a Webots world file with bricks arranged in a grid."
    )
    parser.add_argument("--nx", type=int, required=True, help="Number of bricks in X direction")
    parser.add_argument("--ny", type=int, required=True, help="Number of bricks in Y direction")
    parser.add_argument("--nz", type=int, required=True, help="Number of bricks in Z direction")
    parser.add_argument("--tx", type=float, required=True, help="Translation X for the first brick")
    parser.add_argument("--ty", type=float, required=True, help="Translation Y for the first brick")
    parser.add_argument("--tz", type=float, required=True, help="Translation Z for the first brick")
    parser.add_argument(
        "--template", type=str, required=True, help="Path to the Webots world template file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for the generated Webots world file"
    )
    args = parser.parse_args()

    bricks_placeholder: str = generate_bricks_placeholder(
        args.nx, args.ny, args.nz, (args.tx, args.ty, args.tz)
    )
    process_template(args.template, args.output, bricks_placeholder)


if __name__ == "__main__":
    main()
