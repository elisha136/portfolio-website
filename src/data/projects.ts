/**
 * Projects data file
 * Contains all project information for portfolio
 */

export interface Project {
  title: string;
  description: string;
  image: string;
  category: string;
  status: 'completed' | 'ongoing' | 'planned';
  duration: string;
  role: string;
  institution: string;
  technologies: string[];
  overview: string;
  problemStatement: string;
  challenges: { challenge: string; solution: string; outcome: string }[];
  progress: { task: string; completed: boolean }[];
  github?: string;
  featured?: boolean;
}

export const projectsData: Record<string, Project> = {
  'tarot-t18-drone': {
    title: 'Tarot T18 Octocopter for Autonomous Water Sampling (MSc Thesis)',
    description: 'Building a heavy-lift UAV platform for autonomous water sampling in maritime research, addressing payload limitations identified in the X500 project.',
    image: '/images/projects/tarot-t18-thumb.jpg',
    category: 'UAV Development',
    status: 'ongoing',
    duration: 'September 2024 - Present (MSc Thesis)',
    role: 'Lead Developer',
    institution: 'NTNU IIR Automation Lab',
    technologies: ['ROS2', 'Pixhawk', 'Jetson Orin Nano', 'MAVLink', 'Pozyx UWB', 'Python'],
    overview: 'This MSc thesis project extends the work from the Holybro X500 water sampling drone to a heavy-lift Tarot T18 octocopter platform. The X500 project successfully demonstrated water sampling on the ground but revealed payload constraints (1kg limit) that prevented flight operations with the full sampling system. The T18 platform provides significantly higher payload capacity to accommodate the complete winch system, water containers, and advanced positioning using Pozyx UWB for GPS-denied environments like indoor labs and fjords.',
    problemStatement: 'Traditional water sampling in Norwegian fjords requires boats and significant logistics. The previous X500 project demonstrated the core sampling functionality but could not achieve flight with the payload. This thesis aims to deliver a fully operational autonomous water sampling drone using the heavy-lift Tarot T18 frame with Pozyx UWB positioning for precise indoor/GPS-denied navigation.',
    challenges: [
      { challenge: 'Heavy-lift Platform Integration', solution: 'Migrating validated systems from X500 to T18 frame', outcome: 'In progress - leveraging lessons learned' },
      { challenge: 'Indoor Positioning', solution: 'Integrating Pozyx UWB system with Pixhawk', outcome: 'Currently configuring tag-anchor communication' },
      { challenge: 'Complete Payload Integration', solution: 'Full winch + pump + container system within T18 capacity', outcome: 'Design phase' }
    ],
    progress: [
      { task: 'Frame assembly and motor mounting', completed: true },
      { task: 'Flight controller configuration', completed: true },
      { task: 'Jetson Orin Nano integration', completed: true },
      { task: 'Pozyx UWB positioning', completed: false },
      { task: 'Full payload integration', completed: false },
      { task: 'Autonomous mission planning', completed: false },
      { task: 'Flight testing with payload', completed: false }
    ],
    featured: true
  },
  'ais-collision-prediction': {
    title: 'Maritime Vessel State Prediction Using Machine Learning on AIS Data',
    description: 'Machine learning system for collision risk classification and trajectory prediction achieving 97.54% accuracy on 32.4M AIS records.',
    image: '/images/projects/ais-prediction-thumb.jpg',
    category: 'Machine Learning',
    status: 'completed',
    duration: 'Fall 2025',
    role: 'Developer',
    institution: 'NTNU / Norwegian Electric Systems AS',
    technologies: ['Python', 'Random Forest', 'Pandas', 'Scikit-learn', 'AIS Data', 'NumPy'],
    overview: 'This specialization project developed machine learning models for maritime vessel state prediction using AIS data from Sognefjorden, Norway. Two prediction tasks were addressed: collision risk classification (4 risk levels) and trajectory prediction across multiple time horizons from 30 seconds to 5 minutes. The Random Forest classifier achieved 97.54% test accuracy using 23 engineered features incorporating COLREGS encounter types and maritime domain knowledge.',
    problemStatement: 'Maritime collision avoidance relies heavily on human judgment. By analyzing historical AIS data, we can develop predictive models to assist in early risk identification and improve maritime safety. This project was conducted in collaboration with Norwegian Electric Systems AS to explore integration potential with their RAVEN INS navigation systems.',
    challenges: [
      { challenge: 'Large Dataset Processing', solution: 'Developed efficient data preprocessing pipeline with chunked processing', outcome: 'Successfully processed 32.4M records from 1,065 vessels' },
      { challenge: 'Data Leakage Detection', solution: 'Identified DCPA/TCPA features causing leakage and removed them', outcome: 'Built robust model using 23 kinematic and geometric features' },
      { challenge: 'Feature Engineering', solution: 'Domain-specific features including COLREGS encounter types and relative motion', outcome: 'Achieved 97.54% classification accuracy without explicit CPA calculations' }
    ],
    progress: [
      { task: 'Data collection and preprocessing', completed: true },
      { task: 'Feature engineering with domain knowledge', completed: true },
      { task: 'Collision risk classification model', completed: true },
      { task: 'Trajectory prediction models', completed: true },
      { task: 'Report and documentation', completed: true }
    ],
    featured: true
  },
  'holybro-x500-uav': {
    title: 'Autonomous Water Sampling Drone System',
    description: 'Designed, integrated, and implemented an autonomous water sampling payload for the Holybro X500 V2 quadcopter with Pixhawk flight controller and Jetson Orin Nano companion computer.',
    image: '/images/projects/holybro-x500-thumb.jpg',
    category: 'UAV Development',
    status: 'completed',
    duration: 'Fall 2025',
    role: 'Developer',
    institution: 'NTNU IIR Automation Lab',
    technologies: ['Pixhawk 6C', 'Jetson Orin Nano', 'MAVLink', 'Python', 'GPIO', 'PWM', 'SolidWorks', '3D Printing'],
    overview: 'This MMA4004 Mechatronics and Systems Integration project developed an autonomous water sampling system for the Holybro X500 V2 drone platform. The system integrates mechanical, electrical, and computational subsystems including a peristaltic pump with ±5% volume accuracy for 300ml samples, custom 3D-printed mounting brackets, and MAVLink-based communication between the Pixhawk flight controller and Jetson Orin Nano companion computer. Ground testing successfully demonstrated water sampling functionality with calibration-based volume control.',
    problemStatement: 'Water quality monitoring in remote or hazardous aquatic environments presents significant challenges for traditional boat-based sampling methods. An autonomous aerial platform can reduce sampling time, access difficult-to-reach locations, and provide repeatable, georeferenced sampling points. However, the X500 V2 frame has a maximum payload capacity of only 1kg, which proved insufficient for a complete winch system, leading to scope modifications and the subsequent Tarot T18 thesis project.',
    challenges: [
      { challenge: 'FrSky X8R Receiver Binding Failure', solution: 'Diagnosed incompatible firmware, replaced with X6R receiver with correct ACCESS protocol', outcome: 'Reliable RC communication established' },
      { challenge: 'Pixhawk UART Configuration', solution: 'Systematic debugging of serial port mapping between Jetson and Pixhawk', outcome: 'MAVLink communication over TELEM2 port working' },
      { challenge: 'GPIO PWM Signal Generation', solution: 'Configured pinmux using Jetson-IO tool for proper GPIO output', outcome: 'Achieved 3.3V GPIO output for motor driver control' },
      { challenge: 'Payload Constraints', solution: 'Revised scope to core water sampling demonstration within 1kg limit', outcome: 'Successful ground demo; full system planned for Tarot T18 thesis' }
    ],
    progress: [
      { task: 'Flight platform configuration and calibration', completed: true },
      { task: 'Companion computer integration (Jetson Orin Nano)', completed: true },
      { task: 'MAVLink communication layer', completed: true },
      { task: 'Mechanical design and 3D printing of mounts', completed: true },
      { task: 'Peristaltic pump control with PWM', completed: true },
      { task: 'Volume dispensing calibration (±5% accuracy)', completed: true },
      { task: 'Ground testing and validation', completed: true }
    ],
    featured: true
  },
  'pozyx-uwb-positioning': {
    title: 'Pozyx UWB Indoor Positioning System',
    description: 'Configuration and integration of UWB positioning for GPS-denied navigation.',
    image: '/images/projects/pozyx-uwb-thumb.jpg',
    category: 'Positioning Systems',
    status: 'ongoing',
    duration: 'October 2024 - Present',
    role: 'Developer',
    institution: 'NTNU IIR Automation Lab',
    technologies: ['Pozyx', 'Python', 'MQTT', 'UWB', 'Pixhawk'],
    overview: 'This project involves setting up and calibrating a Pozyx UWB positioning system to enable precise indoor localization for autonomous drones and robots.',
    problemStatement: 'GPS signals are unavailable or unreliable indoors and in many maritime environments. UWB technology provides centimeter-level accuracy for positioning in these GPS-denied scenarios.',
    challenges: [
      { challenge: 'Anchor Placement Optimization', solution: 'Geometric analysis and iterative testing', outcome: 'Optimal coverage with 4 anchors' },
      { challenge: 'Pixhawk Integration', solution: 'Custom MAVLink position messages', outcome: 'In progress' }
    ],
    progress: [
      { task: 'Anchor hardware setup', completed: true },
      { task: 'Tag configuration', completed: true },
      { task: 'Position calculation', completed: true },
      { task: 'Pixhawk integration', completed: false },
      { task: 'Flight testing', completed: false }
    ],
    featured: false
  },
  'robot-kinematics-solver': {
    title: 'Robot Kinematics Solver Using Screw Theory',
    description: 'Developed a comprehensive robot kinematics library implementing forward/inverse kinematics, Jacobian computation, and trajectory planning using screw theory and the Product of Exponentials formula.',
    image: '/images/projects/robot-kinematics-thumb.jpg',
    category: 'Robotics',
    status: 'completed',
    duration: 'Spring 2025',
    role: 'Developer',
    institution: 'NTNU',
    technologies: ['C++', 'Eigen', 'ROS2', 'KDL', 'TracIK', 'Python', 'Screw Theory'],
    overview: 'This AIS4104 Robotics and Intelligent Systems portfolio project (Part 1) involved implementing a complete robot kinematics library from scratch using screw theory and the Product of Exponentials (PoE) formula. The implementation includes forward and inverse kinematics solvers, space and body Jacobian computation, and various trajectory generators for point-to-point and multi-point motions.',
    problemStatement: 'Understanding and implementing robot kinematics is fundamental to robotics. This project required building a comprehensive library that can compute robot configurations, end-effector positions, and plan smooth trajectories, all based on the mathematical foundation of screw theory as presented in Modern Robotics by Lynch and Park.',
    challenges: [
      { challenge: 'Gimbal Lock in Euler Angles', solution: 'Implemented singularity detection and handling for ZYX Euler angle extraction from rotation matrices', outcome: 'Robust angle extraction even near singular configurations' },
      { challenge: 'IK Convergence', solution: 'Implemented Newton-Raphson iteration with damped least squares for numerical stability', outcome: 'Reliable convergence to valid joint configurations' },
      { challenge: 'Trajectory Smoothness', solution: 'Implemented cubic polynomial interpolation with velocity constraints', outcome: 'Smooth trajectories with continuous velocity profiles' }
    ],
    progress: [
      { task: 'Mathematical foundations (rotation matrices, twists, adjoint maps)', completed: true },
      { task: 'Forward kinematics using PoE formula', completed: true },
      { task: 'Inverse kinematics with Newton-Raphson iteration', completed: true },
      { task: 'Space and body Jacobian computation', completed: true },
      { task: 'KDL chain definition for TracIK comparison', completed: true },
      { task: 'Point-to-point trajectory generator', completed: true },
      { task: 'Multi-point trajectory with cubic interpolation', completed: true },
      { task: 'Linear (LIN) trajectory planning', completed: true }
    ],
    featured: true
  },
  'kaya-robot-vision': {
    title: 'NVIDIA Kaya Robot - Computer Vision for Autonomous Navigation',
    description: 'Developed the computer vision system for an autonomous mobile robot using YOLOv8 object detection, Intel RealSense depth sensing, and Extended Kalman Filter for cube detection and 3D positioning.',
    image: '/images/projects/kaya-robot-thumb.jpg',
    category: 'Robotics',
    status: 'completed',
    duration: 'Spring 2025',
    role: 'Computer Vision Lead',
    institution: 'NTNU',
    technologies: ['Python', 'ROS2', 'YOLOv8', 'Intel RealSense D435', 'OpenCV', 'Extended Kalman Filter', 'Jetson Orin Nano'],
    overview: 'This AIS4104 group project (Part 2) involved building and programming an NVIDIA Kaya robot for autonomous cube manipulation. My contribution focused on developing the complete computer vision pipeline: training a YOLOv8 model for cube detection, implementing depth-based 3D positioning using Intel RealSense D435, and integrating an Extended Kalman Filter for stable state estimation. The vision system achieved accurate cube localization enabling the robot to autonomously approach and push cubes to designated positions.',
    problemStatement: 'The Kaya robot needed to autonomously detect colored cubes in its environment, determine their 3D positions, and navigate to manipulate them. This required a robust perception system that could work in real-time with the ROS2-based motion planning and control systems developed by team members.',
    challenges: [
      { challenge: 'Real-time Detection Performance', solution: 'Trained YOLOv8-nano model on custom dataset for optimal speed/accuracy trade-off', outcome: 'Achieved real-time detection at 30+ FPS on Jetson Orin Nano' },
      { challenge: 'Noisy Depth Measurements', solution: 'Implemented Extended Kalman Filter for state estimation and smoothing', outcome: 'Stable 3D position estimates with reduced jitter' },
      { challenge: 'Camera-Robot Coordinate Transform', solution: 'Performed camera calibration and implemented proper coordinate transformations', outcome: 'Accurate cube positions in robot base frame for motion planning' }
    ],
    progress: [
      { task: 'Dataset preparation and annotation', completed: true },
      { task: 'YOLOv8 model training and optimization', completed: true },
      { task: 'Intel RealSense camera integration', completed: true },
      { task: 'Camera calibration for depth accuracy', completed: true },
      { task: 'Depth sensing and 3D positioning', completed: true },
      { task: 'Extended Kalman Filter implementation', completed: true },
      { task: 'ROS2 node development and integration', completed: true },
      { task: 'System testing and validation', completed: true }
    ],
    featured: true
  },
  'blocking-game-ai': {
    title: 'Blocking Game AI Agent',
    description: 'Developed an intelligent game-playing agent for a Blokus-inspired blocking game using Minimax with Alpha-Beta Pruning, Monte Carlo Tree Search, and game-theoretic optimization strategies.',
    image: '/images/projects/blocking-game-thumb.jpg',
    category: 'AI/ML',
    status: 'completed',
    duration: 'Spring 2025',
    role: 'Developer',
    institution: 'NTNU',
    technologies: ['C++', 'Minimax Algorithm', 'Alpha-Beta Pruning', 'Monte Carlo Tree Search', 'Game Theory', 'Transposition Tables'],
    overview: 'This AIS4002 Intelligent Machines Module 1 project involved developing an AI agent to play a competitive turn-based board game similar to Blokus. The agent uses adversarial search algorithms including Minimax with Alpha-Beta Pruning, Monte Carlo Tree Search, and game-theoretic principles to maximize board control while blocking opponents. The implementation features iterative deepening, transposition tables for state caching, and parallel processing for deeper lookahead.',
    problemStatement: 'The Blocking Game is a combinatorial optimization problem where players compete on an N×N grid to place shapes while blocking opponents. The challenge is developing an AI that can evaluate positions, predict opponent moves, and make strategic decisions within strict time constraints (150ms per turn).',
    challenges: [
      { challenge: 'Computational Time Constraints', solution: 'Implemented iterative deepening with time-based cutoff and transposition tables for state caching', outcome: 'Reduced decision time from 150ms to 75ms while maintaining quality' },
      { challenge: 'Large Search Space', solution: 'Combined Alpha-Beta Pruning with move ordering and Monte Carlo fallback', outcome: 'Effectively handled 174+ valid moves per turn' },
      { challenge: 'Multi-Agent Competition', solution: 'Incorporated game-theoretic principles including strategic dominance and opponent modeling', outcome: 'Successfully competed against multiple AI opponents' }
    ],
    progress: [
      { task: 'PEAS definition and environment categorization', completed: true },
      { task: 'Minimax algorithm implementation', completed: true },
      { task: 'Alpha-Beta Pruning optimization', completed: true },
      { task: 'Monte Carlo Tree Search integration', completed: true },
      { task: 'Heuristic function development (edge control, blocking)', completed: true },
      { task: 'Transposition tables and state caching', completed: true },
      { task: 'Iterative deepening search', completed: true },
      { task: 'Performance optimization and testing', completed: true }
    ],
    featured: false
  },
  'qube-servo-reinforcement-learning': {
    title: 'QUBE Servo Pendulum - Deep Reinforcement Learning',
    description: 'Trained reinforcement learning agents to control a Quanser QUBE-Servo 2 inverted pendulum using PPO in Isaac Lab, Soft Actor-Critic (SAC), and Q-Learning with custom Furuta pendulum dynamics simulation.',
    image: '/images/projects/qube-servo-thumb.jpg',
    category: 'AI/ML',
    status: 'completed',
    duration: 'Spring 2025',
    role: 'Developer',
    institution: 'NTNU',
    technologies: ['Python', 'PyTorch', 'Isaac Lab', 'Stable-Baselines3', 'PPO', 'SAC', 'Q-Learning', 'NVIDIA Omniverse', 'ROS'],
    overview: 'This AIS4002 Intelligent Machines Module 2 project explored three different reinforcement learning approaches for the QUBE-Servo 2 pendulum swing-up and stabilization task: (1) PPO training in Isaac Lab with GPU-accelerated parallel simulation, (2) Soft Actor-Critic variants in a custom Python simulator with Lagrangian dynamics, and (3) Q-Learning with discretized state-action spaces. The project included 3D modeling of the QUBE in Fusion 360, URDF/USD conversion, and custom reward function design.',
    problemStatement: 'The Furuta pendulum (inverted rotary pendulum) is a classic nonlinear control problem. The goal is to train an RL agent to swing up the pendulum from its resting position and stabilize it in the inverted (upright) position using only motor torque commands, while handling partial observability and physical constraints.',
    challenges: [
      { challenge: 'Sim-to-Real Transfer Gap', solution: 'Implemented domain randomization and parameter variation during training', outcome: 'Improved robustness though real hardware transfer remains challenging' },
      { challenge: 'Reward Function Design', solution: 'Designed multi-component reward with upright bonus, velocity penalties, and energy shaping', outcome: 'Successful swing-up and stabilization within 3 seconds' },
      { challenge: 'Sample Efficiency', solution: 'Used GPU-accelerated parallel training with 4096 environments in Isaac Lab', outcome: 'Training completed in ~20 minutes with PPO' }
    ],
    progress: [
      { task: '3D modeling and URDF/USD conversion', completed: true },
      { task: 'Isaac Lab environment setup with PPO', completed: true },
      { task: 'Custom Python simulator with Lagrangian dynamics', completed: true },
      { task: 'SAC variants implementation (baseline, frame stacking, RNN)', completed: true },
      { task: 'Q-Learning with discretized state space', completed: true },
      { task: 'Reward function design and optimization', completed: true },
      { task: 'Training and evaluation across methods', completed: true },
      { task: 'TRPO presentation video', completed: true }
    ],
    featured: false
  },
  'computer-vision-detection-segmentation': {
    title: 'Object Detection and Segmentation Pipeline',
    description: 'Built a complete computer vision pipeline for detecting and segmenting household products using YOLOv8, Faster R-CNN, and YOLOv8-Seg, including custom dataset creation, model training, and ONNX deployment.',
    image: '/images/projects/cv-detection-thumb.jpg',
    category: 'AI/ML',
    status: 'completed',
    duration: 'Spring 2025',
    role: 'Developer',
    institution: 'NTNU',
    technologies: ['Python', 'PyTorch', 'YOLOv8', 'Faster R-CNN', 'CVAT', 'Roboflow', 'ONNX', 'Grad-CAM', 'OpenCV'],
    overview: 'This AIS4002 Intelligent Machines Module 3 project developed an end-to-end computer vision pipeline for detecting and segmenting custom household objects (toothpaste, toothbrush, Nivea cream, L\'Oréal Men shower gel, mug, Milo, Nescafé). The project included creating a 143-image custom dataset with polygon annotations, training both one-stage (YOLOv8) and two-stage (Faster R-CNN) detectors, instance segmentation with YOLOv8-Seg, and model interpretation with Grad-CAM visualizations.',
    problemStatement: 'Develop a custom object detection and segmentation system for objects not present in standard benchmark datasets (COCO, LVIS). Compare different detection architectures and deploy the best model in a hardware-agnostic format.',
    challenges: [
      { challenge: 'Small Custom Dataset', solution: 'Used transfer learning from pre-trained models and careful data augmentation', outcome: 'Achieved good detection performance with only 143 images' },
      { challenge: 'Speed vs Accuracy Trade-off', solution: 'Compared YOLOv8 (one-stage) vs Faster R-CNN (two-stage)', outcome: 'YOLOv8 achieved 6× faster inference with minor precision trade-off' },
      { challenge: 'Annotation Quality', solution: 'Used CVAT for precise polygon masks with multiple verification modes', outcome: 'High-quality annotations for both detection and segmentation' }
    ],
    progress: [
      { task: 'Dataset collection (143 images)', completed: true },
      { task: 'Bounding box annotation with LabelImg', completed: true },
      { task: 'Polygon annotation with CVAT', completed: true },
      { task: 'YOLOv8 detection training and evaluation', completed: true },
      { task: 'Faster R-CNN training and evaluation', completed: true },
      { task: 'YOLOv8-Seg instance segmentation', completed: true },
      { task: 'ONNX model conversion for deployment', completed: true },
      { task: 'Grad-CAM visualization and interpretation', completed: true }
    ],
    featured: false
  },
  'fundamentals-automation-mechatronics': {
    title: 'Fundamentals of Automation and Mechatronics',
    description: 'Comprehensive portfolio covering PCB design with NE555 timer circuits, control systems analysis, PLC programming, mechanical engineering calculations, and Python data analysis for weather monitoring.',
    image: '/images/projects/fundamentals-thumb.jpg',
    category: 'Mechatronics',
    status: 'completed',
    duration: 'Fall 2024',
    role: 'Developer',
    institution: 'NTNU',
    technologies: ['Autodesk Fusion 360', 'PCB Design', 'Gerber Files', 'PLC Programming', 'MATLAB', 'Python', 'Control Systems', 'Mechanical Engineering'],
    overview: 'This AIS4003 Fundamentals of Automation and Mechatronics portfolio covered three main modules: (1) Electronics/Mechatronics - designing and soldering a PCB with NE555 timer for audio generation, (2) Mechanical Engineering - robot arm design with DH parameters, torque calculations, servo selection, and deflection analysis, and (3) Automation/Computer Science - PLC programming, control systems transfer functions, and Python data analysis.',
    problemStatement: 'Demonstrate foundational knowledge across the core disciplines of mechatronics engineering: electronics design, mechanical engineering analysis, control systems, industrial automation, and programming.',
    challenges: [
      { challenge: 'PCB Design and Soldering', solution: 'Designed schematic and board layout in Fusion 360, generated Gerber files, and hand-soldered components', outcome: 'Working audio circuit with adjustable frequency via potentiometer' },
      { challenge: 'Robot Arm Calculations', solution: 'Applied DH parameters, inverse kinematics, torque analysis, and deflection calculations', outcome: 'Complete pick-and-place robot design with servo selection' },
      { challenge: 'Control System Analysis', solution: 'Derived transfer functions for DC motor model using Laplace transforms', outcome: 'Mathematical model ready for controller design' }
    ],
    progress: [
      { task: 'PCB schematic design (NE555 timer circuit)', completed: true },
      { task: 'PCB board layout and Gerber file generation', completed: true },
      { task: 'Circuit soldering and testing', completed: true },
      { task: 'Mechanical engineering exercises (stress, deflection)', completed: true },
      { task: 'Robot arm design project', completed: true },
      { task: 'Control systems transfer function derivation', completed: true },
      { task: 'PLC programming portfolio', completed: true },
      { task: 'Python weather data analysis', completed: true }
    ],
    featured: false
  },
  'underwater-sampling-container': {
    title: 'Underwater Sampling Container - Automated Manufacturing',
    description: 'Designed a modular underwater sampling container system for ROV-based marine research, featuring 3D-printed components, automated robotic assembly, and customizable configurations following Industry 4.0/5.0 and DFAA principles.',
    image: '/images/projects/underwater-container-thumb.jpg',
    category: 'Mechatronics',
    status: 'completed',
    duration: 'Fall 2024',
    role: 'Team Member',
    institution: 'NTNU',
    technologies: ['3D Printing', 'CAD Design', 'Collaborative Robots', 'Industry 4.0', 'DFAA', 'Lean Manufacturing', 'PETG Material'],
    overview: 'This MMA4002 Design for Automated Manufacturing group project developed a 3D-printed underwater sampling container system for ROV-based marine research. The design features pressure-resistant containers with customizable sizes, three different clip mechanisms (magnet, twist, slide), and a universal bracket system. The project applied Industry 4.0/5.0 principles, Design for Automated Assembly (DFAA), and Lean Manufacturing concepts, with final assembly performed by collaborative robots.',
    problemStatement: 'Design a customizable, pressure-resistant container system for underwater sample collection that can be automatically manufactured via 3D printing and assembled by collaborative robots, while meeting diverse customer specifications for size, depth rating, and attachment mechanism.',
    challenges: [
      { challenge: 'Pressure Resistance Design', solution: 'Calculated wall thickness based on depth requirements and material properties (PETG)', outcome: 'Containers rated for various ocean depths with appropriate safety factors' },
      { challenge: 'Automated Assembly Optimization', solution: 'Applied DFAA principles to minimize parts and simplify assembly directions', outcome: 'Robot-friendly design suitable for cobot assembly' },
      { challenge: 'Modular Product Architecture', solution: 'Designed interchangeable clips and universal bracket system', outcome: 'Product family with three container sizes and three clip variants' }
    ],
    progress: [
      { task: 'Concept design and workflow planning', completed: true },
      { task: 'Material selection (PETG for underwater use)', completed: true },
      { task: 'Pressure and wall thickness calculations', completed: true },
      { task: 'CAD prototyping and iterations', completed: true },
      { task: 'DFAA evaluation and optimization', completed: true },
      { task: '3D printing of components', completed: true },
      { task: 'Collaborative robot assembly programming', completed: true },
      { task: 'Final product testing and documentation', completed: true }
    ],
    featured: false
  },
  'digital-twins-modal-analysis': {
    title: 'Digital Twin Modal Analysis - Loudspeaker Redesign',
    description: 'Finite element modal analysis of a B&W DM602 S2 loudspeaker enclosure using Siemens NX (SOL103). Successfully shifted the 7th natural frequency above 120 Hz through strategic reinforcement and mass placement.',
    image: '/images/projects/digital-twins-thumb.jpg',
    category: 'Simulation',
    status: 'completed',
    duration: 'Spring 2025',
    role: 'Developer',
    institution: 'NTNU',
    technologies: ['Siemens NX', 'SOL103', 'FEM', 'Modal Analysis', 'Lanczos Eigenvalue Solver', 'CAD'],
    overview: 'This Digital Twins portfolio project applied finite element modal analysis to redesign a loudspeaker cabinet for improved acoustic performance. Using Siemens NX SOL103 with Lanczos eigenvalue extraction, the 7th mode was successfully shifted from 89.27 Hz to 143.56 Hz through internal bracing and strategic mass placement. The project demonstrates skills in Structural Health Monitoring (SHM) and Condition Monitoring (CM) applications.',
    problemStatement: 'Shift the 7th natural vibration mode and all higher modes above 120 Hz to prevent structural resonance effects in critical audible frequencies, improving the loudspeaker\'s acoustic fidelity.',
    challenges: [
      { challenge: 'Mode Below Target Frequency', solution: 'Added internal beam braces and mass elements (0.5kg, 0.3kg, 0.1kg, 0.37kg) at strategic locations', outcome: 'Mode 7 shifted from 89.27 Hz to 143.56 Hz' },
      { challenge: 'Geometry Complexity', solution: 'Simplified CAD model by removing non-essential features while preserving stiffness profile', outcome: 'Reduced computational cost while maintaining modal accuracy' },
      { challenge: 'Material Modeling', solution: 'Used polyethylene as simulation proxy for wood-like behavior', outcome: 'Consistent isotropic material for reliable FEM results' }
    ],
    progress: [
      { task: 'Geometry simplification and import', completed: true },
      { task: 'Material property definition', completed: true },
      { task: '3D tetrahedral mesh generation', completed: true },
      { task: 'SOL103 modal analysis (20 modes)', completed: true },
      { task: 'Reinforcement design and implementation', completed: true },
      { task: 'Post-reinforcement analysis and validation', completed: true }
    ],
    featured: false
  },
  'fishsim-3d-scanning-jig': {
    title: 'FISHSIM - 3D Scanning Jig for Fish Specimens',
    description: 'Designed and prototyped a rotary jig system for comprehensive 3D scanning of fish specimens. This vacation project kickstarted a larger fish research initiative, with the design later modified for actual testing in Tromsø.',
    image: '/images/projects/fishsim-thumb.jpg',
    category: 'Mechatronics',
    status: 'completed',
    duration: 'December 2024',
    role: 'Team Member',
    institution: 'NTNU',
    technologies: ['Fusion 360', 'CAD Design', '3D Printing', 'Stepper Motors', 'Photogrammetry'],
    overview: 'This vacation project developed a jig system with rotary rails for comprehensive 3D scanning of fish specimens. The design features offset rails between the bottom and top base, allowing a camera or scanning device to move seamlessly around the fish from multiple angles. The concept was prototyped in Fusion 360 and built with plans for stepper motor automation. This work kickstarted a larger fish research project, with our design being modified for actual friction testing experiments in Tromsø in October 2025.',
    problemStatement: 'Enable complete and accurate 3D scanning of fish specimens from multiple angles using a portable, adaptable jig system that can accommodate either professional cameras or smartphones.',
    challenges: [
      { challenge: 'Complete Coverage Scanning', solution: 'Designed offset rotary rails allowing camera movement around full specimen circumference', outcome: 'Comprehensive multi-angle scanning capability' },
      { challenge: 'Device Adaptability', solution: 'Created adjustable mounting system for cameras or smartphones', outcome: 'Flexible solution for different scanning scenarios' },
      { challenge: 'Motion Automation', solution: 'Designed for stepper motor integration for smooth, precise rotation', outcome: 'Ready for automated scanning sequences' }
    ],
    progress: [
      { task: 'Conceptual sketching and planning', completed: true },
      { task: 'Fusion 360 CAD modeling', completed: true },
      { task: 'Rail and rotary device design', completed: true },
      { task: 'Prototype fabrication', completed: true },
      { task: 'Initial testing and documentation', completed: true }
    ],
    featured: false
  },
  'autodrone-2025-competition': {
    title: 'AutoDrone 2025 - Autonomous Surface Drone Competition',
    description: 'Represented NTNU at the 2025 AutoDrone competition in Horten. Migrated the codebase from ROS to ROS2, upgraded hardware from Jetson Orin Nano 4GB to 8GB, and implemented autonomous navigation for competition missions.',
    image: '/images/projects/autodrone-thumb.jpg',
    category: 'Robotics',
    status: 'completed',
    duration: 'Spring 2025',
    role: 'Team Member',
    institution: 'NTNU',
    technologies: ['ROS2', 'Python', 'YOLOv8', 'ArduPilot', 'Jetson Orin Nano', 'ZED2i Camera', 'MAVLink', 'GPS'],
    overview: 'This team project continued development of NTNU\'s autonomous sea drone for the 2025 AutoDrone competition in Horten, Norway. Building on a previous bachelor thesis, we migrated the entire codebase from ROS to ROS2, upgraded the onboard computer from Jetson Orin Nano 4GB to 8GB, and implemented mission logic for four competition tasks: Speed Gate, Obstacle Channel, Collision Avoidance, and Visual Docking. At the competition, three of four missions (obstacle avoidance, speed gate, and channel) were completed successfully, with only the docking test experiencing issues.',
    problemStatement: 'Address issues from the previous year\'s competition entry (drone flipped during launch) by improving code structure, upgrading hardware, and implementing reliable autonomous navigation for buoy-based missions.',
    challenges: [
      { challenge: 'ROS to ROS2 Migration', solution: 'Rewrote all launch files, scripts, and communication nodes for ROS2 Humble', outcome: 'Modern, maintainable codebase with improved performance' },
      { challenge: 'YOLO Object Detection', solution: 'Trained and deployed YOLOv8 for real-time buoy detection (red, green, yellow)', outcome: 'Reliable buoy detection for autonomous navigation' },
      { challenge: 'Hardware Upgrade', solution: 'Upgraded from Jetson Orin Nano 4GB to 8GB for better model inference', outcome: 'Faster processing and more stable operation' },
      { challenge: 'Mission Logic Implementation', solution: 'Developed Python scripts for each mission: speed-gate, obstacle-channel, collision-avoidance, docking', outcome: '3 of 4 missions completed successfully at competition' }
    ],
    progress: [
      { task: 'Codebase migration ROS to ROS2', completed: true },
      { task: 'Jetson Orin Nano 8GB upgrade', completed: true },
      { task: 'YOLOv8 model training for buoy detection', completed: true },
      { task: 'Speed gate mission implementation', completed: true },
      { task: 'Obstacle channel mission implementation', completed: true },
      { task: 'Collision avoidance (COLREGs) implementation', completed: true },
      { task: 'Visual docking implementation', completed: true },
      { task: '2025 AutoDrone Competition (May, Horten)', completed: true }
    ],
    github: 'https://github.com/elisha136/AutoDrone2025',
    featured: false
  },
  'talection-skills-measurement': {
    title: 'Talection - Skills Measurement Platform',
    description: 'Contributed to a web application for skills measurement based on the Talection concept. Primarily worked on frontend development using React, TypeScript, and Vite in a team environment.',
    image: '/images/projects/talection-thumb.jpg',
    category: 'Software',
    status: 'completed',
    duration: 'Early 2025',
    role: 'Frontend Developer (Beginner)',
    institution: 'NTNU',
    technologies: ['React', 'TypeScript', 'Vite', 'Vercel', 'REST API', 'Git'],
    overview: 'This collaborative project involved building a web application for skills measurement based on the Talection concept. As someone with limited frontend experience, I contributed primarily to UI component development while learning React and TypeScript. The project was later handed over to computer science students for further development. Despite my limited experience, this project provided valuable lessons in team collaboration, version control workflows, and modern web development practices.',
    problemStatement: 'Create a web-based platform for measuring and tracking skills, providing users with insights into their competencies across different areas.',
    challenges: [
      { challenge: 'Learning New Technologies', solution: 'Studied React and TypeScript while contributing to the project', outcome: 'Gained foundational frontend development skills' },
      { challenge: 'Team Collaboration', solution: 'Used Git, GitHub Projects, and regular communication to coordinate with team members', outcome: 'Learned collaborative development workflows' },
      { challenge: 'Limited Experience', solution: 'Focused on achievable tasks while learning from more experienced team members', outcome: 'Made meaningful contributions within my skill level' }
    ],
    progress: [
      { task: 'Project setup and environment configuration', completed: true },
      { task: 'React component development', completed: true },
      { task: 'API integration work', completed: true },
      { task: 'UI styling and responsiveness', completed: true },
      { task: 'Project handoff to CS students', completed: true }
    ],
    github: 'https://github.com/skcajs/talection-ntnu',
    featured: false
  }
};

export const projectSlugs = Object.keys(projectsData);

