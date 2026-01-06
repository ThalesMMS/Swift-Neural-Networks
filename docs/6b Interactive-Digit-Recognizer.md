# Interactive Digit Recognizer

> **Relevant source files**
> * [README.md](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/README.md)
> * [digit_recognizer.py](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py)

## Purpose and Scope

This document describes the `digit_recognizer.py` Python application, which provides an interactive graphical user interface for testing trained MNIST models. The application allows users to draw handwritten digits on a canvas and receive real-time classification predictions from a previously trained neural network. This tool serves as the inference and deployment component of the system, consuming binary model files produced by the Swift training executables.

For information about model training, see [MNIST MLP Implementation](4a%20MNIST-MLP-Implementation.md). For details on the binary model format consumed by this application, see [Model Binary Format](8%20Model-Binary-Format.md). For training visualization utilities, see [Training Visualization](6a%20Training-Visualization.md).

**Sources:** [Project overview and setup](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/README.md#L200-L212)

 [digit_recognizer.py L1-L14](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L1-L14)

---

## System Architecture

The digit recognizer consists of two primary classes: `MNISTModel` for loading and executing neural network inference, and `DigitRecognizerApp` for managing the Tkinter GUI and user interactions.

### Component Diagram

```mermaid
flowchart TD

MAIN["main() Entry point"]
MODEL_INIT["init() Initialize weights/biases"]
MODEL_LOAD["load_model(filename) Read binary format"]
MODEL_PREDICT["predict(image) Forward pass"]
MODEL_RELU["relu(x) Static activation"]
MODEL_SOFTMAX["softmax(x) Static activation"]
APP_INIT["init(root, model) Initialize GUI"]
APP_SETUP["setup_ui() Create widgets"]
APP_MOUSE_DOWN["on_mouse_down(event) Track start point"]
APP_MOUSE_MOVE["on_mouse_move(event) Draw lines"]
APP_MOUSE_UP["on_mouse_up(event) Trigger prediction"]
APP_CLEAR["clear_canvas() Reset drawing"]
APP_PREDICT["predict() Run inference"]
APP_UPDATE["update_predictions(predictions) Update bars"]
BINARY["mnist_model.bin Binary weights file"]
TK["tkinter.Tk() GUI root window"]
PIL["PIL Image/ImageDraw 28x28 drawing buffer"]

BINARY -.-> MODEL_LOAD
TK -.-> MAIN
PIL -.-> APP_INIT

subgraph digit_recognizer.py ["digit_recognizer.py"]
    MAIN
    MAIN -.-> MODEL_INIT
    MAIN -.-> MODEL_LOAD
    MAIN -.-> APP_INIT

subgraph subGraph1 ["DigitRecognizerApp Class"]
    APP_INIT
    APP_SETUP
    APP_MOUSE_DOWN
    APP_MOUSE_MOVE
    APP_MOUSE_UP
    APP_CLEAR
    APP_PREDICT
    APP_UPDATE
end

subgraph subGraph0 ["MNISTModel Class"]
    MODEL_INIT
    MODEL_LOAD
    MODEL_PREDICT
    MODEL_RELU
    MODEL_SOFTMAX
end
end
```

**Sources:** [digit_recognizer.py L16-L85](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L16-L85)

 [digit_recognizer.py L87-L236](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L87-L236)

 [digit_recognizer.py L238-L290](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L238-L290)

---

## Model Loading Process

The `MNISTModel` class implements binary deserialization of trained model weights. The `load_model` method reads the model file produced by Swift training executables.

### Binary Format Structure

| Offset | Data Type | Description | Size |
| --- | --- | --- | --- |
| 0x00 | `int32` | `num_inputs` (typically 784) | 4 bytes |
| 0x04 | `int32` | `num_hidden` (typically 512) | 4 bytes |
| 0x08 | `int32` | `num_outputs` (typically 10) | 4 bytes |
| 0x0C | `double[]` | Hidden layer weights (row-major: `num_inputs` × `num_hidden`) | 8 × 784 × 512 |
| +offset | `double[]` | Hidden layer biases (`num_hidden`) | 8 × 512 |
| +offset | `double[]` | Output layer weights (row-major: `num_hidden` × `num_outputs`) | 8 × 512 × 10 |
| +offset | `double[]` | Output layer biases (`num_outputs`) | 8 × 10 |

The loading process reads layer dimensions first, allocates NumPy arrays, then unpacks weight matrices row-by-row using the `struct` module with format specifier `'d'` for double-precision floats.

**Code Reference:** [digit_recognizer.py L28-L58](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L28-L58)

### Weight Matrix Organization

```mermaid
flowchart TD

DIMS["Layer Dimensions 3 x int32"]
HW["hidden_weights 784 rows x 512 cols double precision"]
HB["hidden_biases 512 elements double precision"]
OW["output_weights 512 rows x 10 cols double precision"]
OB["output_biases 10 elements double precision"]
NP_HW["self.hidden_weights shape: (784, 512)"]
NP_HB["self.hidden_biases shape: (512,)"]
NP_OW["self.output_weights shape: (512, 10)"]
NP_OB["self.output_biases shape: (10,)"]

HW -.->|"struct.unpack"| NP_HW
HB -.-> NP_HB
OW -.-> NP_OW
OB -.-> NP_OB

subgraph subGraph1 ["NumPy Arrays in MNISTModel"]
    NP_HW
    NP_HB
    NP_OW
    NP_OB
end

subgraph subGraph0 ["Memory Layout in mnist_model.bin"]
    DIMS
    HW
    HB
    OW
    OB
    DIMS -.-> HW
    HW -.->|"struct.unpack row-major"| HB
    HB -.->|"struct.unpack"| OW
    OW -.->|"struct.unpack row-major"| OB
end
```

**Sources:** [digit_recognizer.py L28-L58](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L28-L58)

 README.md

---

## Inference Pipeline

The `predict` method implements a forward pass through the two-layer MLP architecture without gradient computation. This matches the architecture defined in the Swift training code.

### Forward Pass Computation

```mermaid
flowchart TD

INPUT["Input: img_flat shape: (784,) range: [0, 1]"]
MATMUL1["np.dot(image, hidden_weights) + hidden_biases Result shape: (512,)"]
RELU["relu(hidden) np.maximum(0, x) Result shape: (512,)"]
MATMUL2["np.dot(hidden, output_weights) + output_biases Result shape: (10,)"]
SOFTMAX["softmax(output) exp(x) / sum(exp(x)) Result shape: (10,)"]
RESULT["Output: probabilities 10 class confidences sum = 1.0"]

INPUT -.-> MATMUL1
RELU -.-> MATMUL2
SOFTMAX -.-> RESULT

subgraph subGraph1 ["Output Layer Computation"]
    MATMUL2
    SOFTMAX
    MATMUL2 -.-> SOFTMAX
end

subgraph subGraph0 ["Hidden Layer Computation"]
    MATMUL1
    RELU
    MATMUL1 -.-> RELU
end
```

**Mathematical Formulation:**

1. **Hidden Layer:** `hidden = ReLU(image · W₁ + b₁)` where `W₁ ∈ ℝ^(784×512)`, `b₁ ∈ ℝ^512`
2. **Output Layer:** `logits = hidden · W₂ + b₂` where `W₂ ∈ ℝ^(512×10)`, `b₂ ∈ ℝ^10`
3. **Probabilities:** `p = softmax(logits)` where `p[i] = exp(logits[i]) / Σⱼ exp(logits[j])`

**Sources:** [digit_recognizer.py L71-L84](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L71-L84)

 [digit_recognizer.py L60-L69](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L60-L69)

---

## GUI Components

The `DigitRecognizerApp` class creates a Tkinter interface with three primary regions: drawing canvas, prediction display, and control buttons.

### Layout Structure

```mermaid
flowchart TD

BTN_CLEAR["ttk.Button 'Clear' → clear_canvas()"]
BTN_PRED["ttk.Button 'Predict' → predict()"]
BAR0["Digit 0: Canvas(150x20) + Label"]
BAR1["Digit 1: Canvas(150x20) + Label"]
BARDOT["..."]
BAR9["Digit 9: Canvas(150x20) + Label"]
PRED_TITLE["ttk.Label 'Predictions:'"]
TITLE["ttk.Label 'Draw a digit (0-9)' Font: Arial 16 bold"]
CANVAS["tk.Canvas width=280, height=280 bg='black', cursor='cross' Bindings: Button-1, B1-Motion, ButtonRelease-1"]
INSTR["ttk.Label Instructions text foreground='gray'"]
PIL_IMAGE["PIL.Image 28x28 grayscale Internal buffer"]
PIL_DRAW["PIL.ImageDraw Drawing context"]

CANVAS -.->|"Mouse events scaled coordinates"| PIL_DRAW

subgraph subGraph4 ["Tkinter Root Window"]
    TITLE
    INSTR

subgraph subGraph0 ["Left Column: Drawing Area"]
    CANVAS
end

subgraph subGraph3 ["Button Row"]
    BTN_CLEAR
    BTN_PRED
end

subgraph subGraph2 ["Right Column: Prediction Panel"]
    PRED_TITLE

subgraph subGraph1 ["10 Prediction Rows"]
    BAR0
    BAR1
    BARDOT
    BAR9
end
end
end
```

**Key Instance Variables:**

| Variable | Type | Purpose |
| --- | --- | --- |
| `self.canvas` | `tk.Canvas` | 280×280 pixel display canvas for user drawing |
| `self.image` | `PIL.Image` | 28×28 grayscale buffer (internal representation) |
| `self.draw` | `PIL.ImageDraw` | Drawing context for PIL image |
| `self.pred_bars` | `list[tk.Canvas]` | 10 bar canvases for visualization |
| `self.pred_labels` | `list[ttk.Label]` | 10 labels showing percentage values |
| `self.last_x`, `self.last_y` | `int` | Previous mouse coordinates for line drawing |

**Sources:** [digit_recognizer.py L90-L170](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L90-L170)

 [digit_recognizer.py L96-L101](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L96-L101)

 [digit_recognizer.py L137-L154](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L137-L154)

---

## Drawing and Prediction Workflow

The application implements real-time drawing with automatic prediction triggering. Mouse events are processed to draw on both the display canvas and the internal 28×28 buffer simultaneously.

### Event Processing Flow

```mermaid
sequenceDiagram
  participant p1 as User
  participant p2 as tk.Canvas (280x280)
  participant p3 as Event Handlers
  participant p4 as PIL.Image (28x28)
  participant p5 as MNISTModel
  participant p6 as Prediction Bars

  p1->>p2: Press mouse button
  p2->>p3: on_mouse_down(event)
  p3->>p3: Store event.x, event.y in last_x, last_y
  note over p1,p6: Drawing Loop
  p1->>p2: Drag mouse (B1-Motion)
  p2->>p3: on_mouse_move(event)
  p3->>p2: create_line(last_x | last_y | event.x | event.y | fill='white' | width=20)
  p3->>p3: Scale coordinates: scale = 28 / 280 x_img = int(event.x * scale)
  p3->>p4: draw.line([x1_img | y1_img | x2_img | y2_img] | fill=255 | width=scaled_brush)
  p3->>p3: Update last_x, last_y
  p1->>p2: Release mouse button
  p2->>p3: on_mouse_up(event)
  p3->>p3: Clear last_x, last_y
  p3->>p3: Auto-trigger predict()
  p3->>p4: Convert to numpy array np.array(image) / 255.0
  p3->>p4: Flatten to 784 elements
  p3->>p5: predict(img_flat)
  p5->>p5: Forward pass: hidden + output layers
  p5-->>p3: Return probabilities (10 |)
  p3->>p6: update_predictions(predictions)
  loop For each digit 0-9
    p6->>p6: Clear bar canvas
    p6->>p6: Draw rectangle: width = prob * 140 color = green if argmax else blue
    p6->>p6: Update label text: f"{prob*100:.1f}%"
  end
  p6-->>p1: Visual feedback
```

**Coordinate Scaling:** The display canvas operates at 280×280 pixels for better visibility, while the internal PIL image maintains MNIST's native 28×28 resolution. The scaling factor `scale = 28 / 280 = 0.1` is applied to mouse coordinates before drawing to the internal buffer.

**Sources:** [digit_recognizer.py L171-L204](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L171-L204)

 [digit_recognizer.py L213-L223](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L213-L223)

 [digit_recognizer.py L225-L235](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L225-L235)

---

## Data Preprocessing

Before inference, the drawn image undergoes normalization to match the training data distribution.

### Preprocessing Pipeline

```mermaid
flowchart TD

DRAW["PIL.Image 28x28 pixels uint8 [0, 255]"]
NUMPY["np.array(image) shape: (28, 28) dtype: uint8"]
NORM["image / 255.0 shape: (28, 28) dtype: float32 range: [0.0, 1.0]"]
FLAT["flatten() shape: (784,) dtype: float32 range: [0.0, 1.0]"]

DRAW -.-> NUMPY
NUMPY -.-> NORM
NORM -.-> FLAT
```

**Implementation:** [digit_recognizer.py L214-L217](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L214-L217)

```
img_array = np.array(self.image).astype(np.float32) / 255.0img_flat = img_array.flatten()
```

This preprocessing matches the normalization applied during training, where MNIST pixel values (originally uint8 in [0, 255]) are divided by 255.0 to produce float values in [0.0, 1.0].

**Sources:** [digit_recognizer.py L213-L223](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L213-L223)

---

## Command-Line Interface

The application accepts an optional command-line argument specifying which model file to load.

### Model File Selection

```mermaid
flowchart TD

START["main() execution"]
ARGS["sys.argv length?"]
PARSE["Parse sys.argv[1] as model_type"]
SEARCH["Search for available models"]
VALIDATE["model_type in model_files dict?"]
SELECTED["model_file = model_files[model_type]"]
ERROR1["Print usage and exit"]
LOOP["Loop through model_files dict"]
CHECK["os.path.exists(mfile)?"]
NEXT["Try next file"]
ERROR2["Print error and exit"]
LOAD["model.load_model(model_file)"]
GUI["Create Tkinter GUI DigitRecognizerApp"]
RUN["root.mainloop()"]

START -.->|"len > 1"| ARGS
ARGS -.-> PARSE
ARGS -.->|"len == 1"| SEARCH
PARSE -.->|"Found"| VALIDATE
VALIDATE -.->|"Yes"| SELECTED
VALIDATE -.->|"No"| ERROR1
SEARCH -.-> LOOP
LOOP -.->|"Not found"| CHECK
CHECK -.-> SELECTED
CHECK -.->|"None found"| NEXT
NEXT -.-> CHECK
CHECK -.-> ERROR2
SELECTED -.-> LOAD
LOAD -.-> GUI
GUI -.-> RUN
```

**Model File Mapping:**

| Argument | File Name | Description |
| --- | --- | --- |
| `serial` | `mnist_model.bin` | Default model from Swift training |
| `cpu` | `mnist_model_cpu.bin` | Alternative CPU model |
| `gpu` | `mnist_model_gpu.bin` | Alternative GPU model |
| `cuda` | `mnist_model_gpu.bin` | CUDA model (same as GPU) |

**Usage Examples:**

```
# Load default model (searches for any available model)python digit_recognizer.py# Load specific model typepython digit_recognizer.py serialpython digit_recognizer.py gpu
```

**Sources:** [digit_recognizer.py L238-L286](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L238-L286)

 [digit_recognizer.py L240-L246](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L240-L246)

---

## Prediction Display

The prediction panel displays class probabilities as horizontal bar charts with percentage labels. The predicted class (highest probability) is highlighted with a distinct color.

### Bar Visualization Logic

```mermaid
flowchart TD

START["update_predictions(predictions) Input: numpy array shape (10,)"]
LOOP["For i in range(10)"]
COMPUTE["Compute bar_width: int(predictions[i] * 140)"]
CLEAR["pred_bars[i].delete('all') Clear previous bar"]
ARGMAX["i == np.argmax(predictions)?"]
COLOR1["color = '#4CAF50' (green)"]
COLOR2["color = '#2196F3' (blue)"]
DRAW["pred_bars[i].create_rectangle (0, 0, bar_width, 20, fill=color)"]
LABEL["pred_labels[i].config text=f'{predictions[i]*100:.1f}%'"]
NEXT["Continue to next digit"]
END["Display complete"]

START -.-> LOOP
LOOP -.-> COMPUTE
COMPUTE -.-> CLEAR
CLEAR -.-> ARGMAX
ARGMAX -.->|"Yes (highest prob)"| COLOR1
ARGMAX -.->|"No"| COLOR2
COLOR1 -.-> DRAW
COLOR2 -.-> DRAW
DRAW -.-> LABEL
LABEL -.-> NEXT
NEXT -.-> LOOP
LOOP -.-> END
```

**Bar Width Calculation:** Each bar canvas has a width of 150 pixels, but the rectangle width is computed as `int(predictions[i] * 140)`, leaving a 10-pixel margin. This means a probability of 1.0 (100%) displays as a 140-pixel bar.

**Color Coding:**

* **Green (`#4CAF50`):** Predicted class (highest probability)
* **Blue (`#2196F3`):** All other classes

**Sources:** [digit_recognizer.py L225-L235](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L225-L235)

---

## Technical Implementation Details

### Canvas Drawing Parameters

| Parameter | Value | Purpose |
| --- | --- | --- |
| `canvas_size` | 280 | Display canvas dimensions (pixels) |
| `brush_size` | 20 | Brush diameter for display canvas (pixels) |
| PIL image size | 28×28 | Native MNIST resolution |
| PIL brush size | `max(1, int(brush_size * scale))` | Scaled brush for internal buffer |
| Scale factor | 0.1 | Ratio of PIL size to display size (28/280) |

### Mouse Event Bindings

| Event | Handler | Description |
| --- | --- | --- |
| `<Button-1>` | `on_mouse_down` | Left mouse button press |
| `<B1-Motion>` | `on_mouse_move` | Mouse movement while button held |
| `<ButtonRelease-1>` | `on_mouse_up` | Left mouse button release |

The application uses `capstyle=tk.ROUND` for line drawing to create smooth, continuous strokes rather than segmented rectangular lines.

**Sources:** [digit_recognizer.py L95-L108](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L95-L108)

 [digit_recognizer.py L126-L128](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L126-L128)

 [digit_recognizer.py L183-L194](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L183-L194)

---

## Dependencies

The application requires the following Python packages:

| Package | Purpose |
| --- | --- |
| `numpy` | Numerical operations, array manipulation, inference computation |
| `tkinter` | GUI framework (standard library, but may require separate install on some systems) |
| `Pillow` (PIL) | Image creation, drawing primitives, pixel buffer management |

**Installation:**

```
pip install -r requirements.txt
```

**Sources:** [Project overview and setup](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/README.md#L208-L212)

 [digit_recognizer.py L7-L13](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L7-L13)

---

## Integration with Training Pipeline

The digit recognizer serves as the final stage in the machine learning workflow, consuming artifacts produced by the Swift training executables.

### Data Flow Integration

```mermaid
flowchart TD

SWIFT["mnist_mlp_swift or other trainer"]
TRAIN["Training process: Forward/backward passes Weight updates"]
SAVE["Serialize model: layer dims + weights"]
LOAD["digit_recognizer.py MNISTModel.load_model()"]
GUI["DigitRecognizerApp Tkinter interface"]
INFER["predict() method Forward pass only"]
MODEL["mnist_model.bin Binary format Double precision"]
USER["User draws digit"]
RESULT["Classification result"]

SAVE -.-> MODEL
MODEL -.-> LOAD
USER -.-> GUI
INFER -.-> RESULT

subgraph subGraph1 ["Inference Stage (Python)"]
    LOAD
    GUI
    INFER
    LOAD -.-> GUI
    GUI -.-> INFER
end

subgraph subGraph0 ["Training Stage (Swift)"]
    SWIFT
    TRAIN
    SAVE
    SWIFT -.-> TRAIN
    TRAIN -.-> SAVE
end
```

The binary model file acts as the interface contract between training and inference systems. The model must be trained before the digit recognizer can function. If no model file exists, the application displays an error message and exits gracefully.

**Sources:** [digit_recognizer.py L238-L286](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/digit_recognizer.py#L238-L286)

 **Sources**: [Project overview and setup](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/README.md#L200-L212)





### On this page

* [Interactive Digit Recognizer](#6.2-interactive-digit-recognizer)
* [Purpose and Scope](#6.2-purpose-and-scope)
* [System Architecture](#6.2-system-architecture)
* [Component Diagram](#6.2-component-diagram)
* [Model Loading Process](#6.2-model-loading-process)
* [Binary Format Structure](#6.2-binary-format-structure)
* [Weight Matrix Organization](#6.2-weight-matrix-organization)
* [Inference Pipeline](#6.2-inference-pipeline)
* [Forward Pass Computation](#6.2-forward-pass-computation)
* [GUI Components](#6.2-gui-components)
* [Layout Structure](#6.2-layout-structure)
* [Drawing and Prediction Workflow](#6.2-drawing-and-prediction-workflow)
* [Event Processing Flow](#6.2-event-processing-flow)
* [Data Preprocessing](#6.2-data-preprocessing)
* [Preprocessing Pipeline](#6.2-preprocessing-pipeline)
* [Command-Line Interface](#6.2-command-line-interface)
* [Model File Selection](#6.2-model-file-selection)
* [Prediction Display](#6.2-prediction-display)
* [Bar Visualization Logic](#6.2-bar-visualization-logic)
* [Technical Implementation Details](#6.2-technical-implementation-details)
* [Canvas Drawing Parameters](#6.2-canvas-drawing-parameters)
* [Mouse Event Bindings](#6.2-mouse-event-bindings)
* [Dependencies](#6.2-dependencies)
* [Integration with Training Pipeline](#6.2-integration-with-training-pipeline)
* [Data Flow Integration](#6.2-data-flow-integration)

Ask Devin about Swift-Neural-Networks