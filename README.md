# Multi-DOF Bionic Prosthetic Hand Based on RDX X5

Our team has designed a multi-degree-of-freedom (DOF) bionic prosthetic hand based on the RDX X5 platform. It features a 16-DOF hand and a 5-DOF robotic arm, aiming to achieve true multi-DOF functionality. Our goal is to solve the daily life challenges of amputees成本-effectively, serve as a crucial component for embodied intelligence through its portability, and act as a capable assistant in the medical and disaster relief fields with its remote operation capabilities.

## Core Features

*   **High Degrees of Freedom:** A 16-DOF hand combined with a 5-DOF arm enables highly sophisticated and life-like movements.
*   **Cost-Effective Solution:** Designed to provide an affordable yet powerful prosthetic option for the disabled community.
*   **High Portability:** Easily adaptable and integrable into various embodied AI systems.
*   **Remote Operation:** Supports remote control for applications in telemedicine and disaster response scenarios.

## System Architecture

Our embedded intelligent prosthetic system follows a layered design philosophy, comprising four core layers: Perception, Control, Software, and Hardware.

<img width="1471" height="1314" alt="System Architecture Diagram" src="https://github.com/user-attachments/assets/ec4f5e5f-1579-419e-827b-d41e22011098" />

*Figure 1: System Components of the Intelligent Prosthetic Hand*

### 1. Perception Layer

The Perception Layer is responsible for collecting multi-modal information from the external environment and converting it into a four-dimensional information network (vision, audio, motor torque, and motor current) for the Control Layer's decision-making.

*   **Visual Perception:** Utilizes a YOLO model to analyze color and depth images from the camera, outputting 3D detection data, object categories, point clouds, and more.
*   **Auditory Perception:** A locally deployed KWS (Keyword Spotting) model detects wake words with low computational overhead.
*   **Haptic/Current Feedback:** Reads motor torque and current data for fine-grained control and safety assurance.

### 2. Control Layer

The Control Layer serves as the brain of the system, responsible for decision-making and coordinating all components. We employ an advanced robotics operating system and simulation platform to achieve high-precision motor control.

*   **ROS 2 (Robot Operating System 2):** Acts as the core middleware, bridging the Perception, Software, and Hardware layers for data communication and task scheduling.
    *   Custom ROS nodes publish perception data.
    *   The Software and Hardware layers subscribe to relevant topics to receive information.
    *   The `ros2_control` framework manages real-time motor position data and future motor positions inferred by the network.
*   **Isaac Gym Simulation Platform:** Used to train the policy network that controls motor behavior.
    *   Training is conducted using the A2C (Advantage Actor-Critic) reinforcement learning algorithm.
    *   Adversarial training with an AMP (Adversarial Motion Priors) discriminator is used to enhance the accuracy and coherence of the agent's actions.
    *   The trained model enables the robot to adapt to various environments.

### 3. Software Layer

The Software Layer integrates various functional calls to enhance the intelligence and user experience of the prosthetic hand.

*   **Motor Debugging:** Dynamixel Wizard
*   **Model Customization:** Coze Platform, KWS Platform
*   **Edge Large Model Gateway:** Volcano Engine Ark
*   **Cloud Data Processing:** Volcano Engine IoT Platform

Through these software components, we have implemented:

*   **Mobile App Control:** Users can interact with the prosthetic hand via a mobile application.
*   **Cloud Data Processing:** Utilizes cloud platforms for data analysis and model optimization.
*   **Edge AI Agent Invocation:** Enables localized intelligent responses and computations.

#### Cloud-Edge Communication and Voice Interaction

*   **KWS Activation & Node.js Server:** A C++ node receives the KWS activation signal and passes it to a Node.js server.
*   **Volcano Engine Ark Integration:** The Node.js server calls the Volcano Engine `StartVoiceChat` OpenAPI, configuring ASR (Automatic Speech Recognition), TTS (Text-to-Speech), and the Ark LLM (Large Language Model).
*   **Real-Time Communication:** Generates connection parameters for the RTC client, which are then passed back to the C++ client to join the session for real-time cloud-edge communication.
*   **AI Response & Playback:** Callback functions display the AI-generated text response and play it back as audio.

### 4. Hardware Layer

The Hardware Layer forms the physical foundation of the system, responsible for executing commands and interacting with the environment.

*   **Embedded Development Board:** The RDK X5 serves as the "brain" of the project, deploying and coordinating all functions from the Perception and Control layers.
*   **Robotic Arm and Hand:** As the "body" of the project, it receives and executes commands from the RDK X5.

#### Edge Control and Communication

*   **WebSocket Bridge Node:** A `websocket_bridge_node` is set up on the RDK X5 to subscribe to the YOLO node's output.
*   **App Communication Bridge:** This node acts as a bridge between the ROS system and external applications, forwarding text commands from the app to ROS topics and broadcasting key information (like object detection results and task status) to all connected app clients.
*   **HBuilder Mobile App:** A dedicated mobile app developed with HBuilder provides convenient control from the edge device.

## Technology Stack

*   **Hardware:** RDK X5, Robotic Arm (16-DOF Hand, 5-DOF Arm)
*   **Operating System:** ROS 2
*   **Simulation:** Isaac Gym
*   **Reinforcement Learning:** A2C, AMP
*   **Perception:** YOLO, KWS
*   **Communication:** ROS 2 Topics, WebSocket
*   **Cloud Services:** Volcano Engine (IoT Platform, Ark Engine)
*   **App Development:** HBuilder

<img width="1706" height="1279" alt="Project Demonstration" src="https://github.com/user-attachments/assets/010688d3-7bbe-4be8-a731-f4a557b53628" />
