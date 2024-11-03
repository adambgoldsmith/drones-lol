import * as THREE from 'three';
import { PointerLockControls } from 'https://cdn.jsdelivr.net/npm/three@0.146.0/examples/jsm/controls/PointerLockControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';


// Scene setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Player movement variables
const moveSpeed = 0.15;
const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
let moveForward = false;
let moveBackward = false;
let moveLeft = false;
let moveRight = false;
let isDrawing = false;
let currentCanvas = null;
let isInteracting = false;

// Controls setup
const controls = new PointerLockControls(camera, document.body);

// Room dimensions
const roomWidth = 20;
const roomHeight = 8;
const roomDepth = 15;

// Canvas dimensions
const smallCanvasWidth = 2;
const smallCanvasHeight = 2;
const bigCanvasWidth = 6;
const bigCanvasHeight = 4;

// Create room
const roomGeometry = new THREE.BoxGeometry(roomWidth, roomHeight, roomDepth);
const roomMaterial = new THREE.MeshBasicMaterial({ 
    color: 0xcccccc, 
    side: THREE.BackSide 
});
const room = new THREE.Mesh(roomGeometry, roomMaterial);
scene.add(room);

// Create model loader
const loader = new GLTFLoader();

// Create radio for music
const radio = new THREE.Group();
loader.load('models/radio.glb', function(gltf) {
    radio.add(gltf.scene);
    // place radio in back corner of room
    radio.position.set(roomWidth/2 - 1, 0.5, roomDepth/2 - 1);
    radio.scale.set(0.25, 0.25, 0.25);
    // rotate radio to face into room
    radio.rotation.set(0, Math.PI + Math.PI/4, 0);
    scene.add(radio);
});

// Add lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(0, 1, 0);
scene.add(directionalLight);

// positional audio
const listener = new THREE.AudioListener();
camera.add(listener);
const positionalSound = new THREE.PositionalAudio(listener);
const audioLoader = new THREE.AudioLoader();
audioLoader.load('elevator_bossa_nova.mp3', function(buffer) {
    positionalSound.setBuffer(buffer);
    positionalSound.setRefDistance(1);
    positionalSound.setLoop(true);
    positionalSound.setVolume(0.5);
    positionalSound.play();
});
radio.add(positionalSound);

// Add paintbrush model in front of camera
scene.add(camera);
const paintbrush = new THREE.Group();
loader.load('models/paintbrush.glb', function(gltf) {
    paintbrush.add(gltf.scene);
    camera.add(paintbrush);
    // position paintbrush in front of camera
    paintbrush.scale.set(0.1, 0.1, 0.1);
    paintbrush.rotation.set(-Math.PI/4, 0, 0);
    paintbrush.position.set(0.5, -0.5, -0.5);
});

// Create drawing canvases
const canvases = [];
const drawingContexts = [];
const canvasTextures = [];
const drawingData = new Array(4).fill(null);

function createCanvas(width, height, position, rotation) {
    // Create HTML canvas for drawing
    const drawingCanvas = document.createElement('canvas');
    drawingCanvas.width = 512;
    drawingCanvas.height = 512;
    const ctx = drawingCanvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 512, 512);
    
    // Create Three.js canvas
    const texture = new THREE.CanvasTexture(drawingCanvas);
    const geometry = new THREE.PlaneGeometry(width, height);
    const material = new THREE.MeshBasicMaterial({ map: texture });
    const canvas = new THREE.Mesh(geometry, material);
    
    canvas.position.set(...position);
    canvas.rotation.set(...rotation);
    
    scene.add(canvas);
    canvases.push(canvas);
    drawingContexts.push(ctx);
    canvasTextures.push(texture);
    
    return canvas;
}

// Create small canvases on opposite walls
createCanvas(smallCanvasWidth, smallCanvasHeight, [-roomWidth/2 + 0.01, 2, -3], [0, Math.PI/2, 0]);
createCanvas(smallCanvasWidth, smallCanvasHeight, [-roomWidth/2 + 0.01, 2, 3], [0, Math.PI/2, 0]);
createCanvas(smallCanvasWidth, smallCanvasHeight, [roomWidth/2 - 0.01, 2, -3], [0, -Math.PI/2, 0]);
createCanvas(smallCanvasWidth, smallCanvasHeight, [roomWidth/2 - 0.01, 2, 3], [0, -Math.PI/2, 0]);

// Create main canvas
const mainCanvas = createCanvas(bigCanvasWidth, bigCanvasHeight, [0, 3, -roomDepth/2 + 0.01], [0, 0, 0]);

// Create button under main canvas
const buttonGeometry = new THREE.BoxGeometry(1, 0.5, 0.1);
const buttonMaterial = new THREE.MeshBasicMaterial({ color: 0x4444ff });
const button = new THREE.Mesh(buttonGeometry, buttonMaterial);
button.position.set(0, 1, -roomDepth/2 + 0.1);
scene.add(button);

// Set initial camera position
camera.position.set(0, 2, roomDepth/2 - 2);

// Event listeners
document.addEventListener('click', function() {
    if (!controls.isLocked) {
        controls.lock();
    }
});

document.addEventListener('mousedown', function() {
    if (currentCanvas !== null && controls.isLocked) {
        isDrawing = true;
    }
    isInteracting = true;
});

document.addEventListener('mouseup', function() {
    isDrawing = false;
    isInteracting = false;
    if (currentCanvas !== null) {
        // Here you would process the drawing with ML model
        // drawingData[currentCanvas] = processDrawingWithML(drawingContexts[currentCanvas]);
    }
});

document.addEventListener('keydown', function(event) {
    switch (event.code) {
        case 'ArrowUp':
        case 'KeyW':
            moveForward = true;
            break;
        case 'ArrowDown':
        case 'KeyS':
            moveBackward = true;
            break;
        case 'ArrowLeft':
        case 'KeyA':
            moveLeft = true;
            break;
        case 'ArrowRight':
        case 'KeyD':
            moveRight = true;
            break;
    }
});

document.addEventListener('keyup', function(event) {
    switch (event.code) {
        case 'ArrowUp':
        case 'KeyW':
            moveForward = false;
            break;
        case 'ArrowDown':
        case 'KeyS':
            moveBackward = false;
            break;
        case 'ArrowLeft':
        case 'KeyA':
            moveLeft = false;
            break;
        case 'ArrowRight':
        case 'KeyD':
            moveRight = false;
            break;
    }
});

// Raycaster for interaction detection
const raycaster = new THREE.Raycaster();
const interactionDistance = 3;

function checkInteractions() {
    // Direction player is facing
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    
    raycaster.set(camera.position, direction);
    
    // Check for radio interactions
    const radioIntersects = raycaster.intersectObject(radio);
    if (radioIntersects.length > 0 && radioIntersects[0].distance < interactionDistance) {
        if (isInteracting) {
            if (positionalSound.isPlaying) {
                positionalSound.pause();
            } else {
                positionalSound.play();
            }
            isInteracting = false;
        }
    }

    // Check for canvas interactions
    const intersects = raycaster.intersectObjects(canvases);
    
    if (intersects.length > 0 && intersects[0].distance < interactionDistance) {
        currentCanvas = canvases.indexOf(intersects[0].object);
        
        // Handle drawing if mouse is pressed
        if (isDrawing) {
            const intersect = intersects[0];
            const point = intersect.uv;
            if (point) {
                const ctx = drawingContexts[currentCanvas];
                ctx.fillStyle = 'black';
                ctx.beginPath();
                ctx.arc(
                    point.x * 512,
                    (1 - point.y) * 512,
                    5,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
                canvasTextures[currentCanvas].needsUpdate = true;
            }
        }
    } else {
        currentCanvas = null;
    }
    
    // Check for button interaction
    const buttonIntersects = raycaster.intersectObject(button);
    if (buttonIntersects.length > 0 && buttonIntersects[0].distance < interactionDistance) {
        button.material.color.setHex(0x6666ff); // Highlight button
        if (controls.isLocked && isDrawing) {
            // Here you would process all drawings and generate the final image
            // const keywords = drawingData.filter(data => data !== null);
            // const finalImage = await generateImageWithAI(keywords);
            // Update main canvas with finalImage
        }
    } else {
        button.material.color.setHex(0x4444ff);
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    if (controls.isLocked) {
        // Update velocity based on movement keys
        direction.z = Number(moveForward) - Number(moveBackward);
        direction.x = Number(moveRight) - Number(moveLeft);
        direction.normalize();
        
        velocity.z = direction.z * moveSpeed;
        velocity.x = direction.x * moveSpeed;
        
        controls.moveRight(velocity.x);
        controls.moveForward(velocity.z);
        
        // Keep player within room bounds
        camera.position.y = 2; // Lock height
        camera.position.x = Math.max(-roomWidth/2 + 1, Math.min(roomWidth/2 - 1, camera.position.x));
        camera.position.z = Math.max(-roomDepth/2 + 1, Math.min(roomDepth/2 - 1, camera.position.z));
    }
    
    checkInteractions();
    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', function() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();