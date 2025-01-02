//
//  ViewController.swift
//  ConcurrencyiOS
//
//  Created by BJIT on 23/12/24.
//

import UIKit
import MetalKit
import Vision
import AVFoundation
import CoreImage

class MLMetalViewController: UIViewController {
    private var metalView: MTKView!
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var pipelineState: MTLRenderPipelineState!
    private var captureSession: AVCaptureSession!
    private var videoOutput: AVCaptureVideoDataOutput!

    // ML Properties
    private var detectionRequest: VNCoreMLRequest!
    private var visionModel: VNCoreMLModel!
    private var detectedObjects: [VNRecognizedObjectObservation] = []
    private var vertexBuffer: MTLBuffer!
    private var texture: MTLTexture?
    private var samplerState: MTLSamplerState!
    private var textureCache: CVMetalTextureCache?

    // Vertices for a full-screen quad
    private let vertices: [Float] = [
        -1.0, -1.0, 0.0, 1.0,
         1.0, -1.0, 0.0, 1.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 0.0, 1.0,
    ]



    // Uniform buffer for passing detection data to Metal
    struct Uniforms {
        var numObjects: UInt32
        var padding: SIMD3<UInt32>
        var objectBoxes: [SIMD4<Float>]

        init() {
            numObjects = 0
            padding = SIMD3<UInt32>(0, 0, 0)
            objectBoxes = Array(repeating: SIMD4<Float>(0, 0, 0, 0), count: 20)
        }
    }


    private var uniformBuffer: MTLBuffer!
    private var uniforms = Uniforms()

    override func viewDidLoad() {
        super.viewDidLoad()
        setupMetal()
        setupTextureCache() // Add this new setup
        setupML()
        setupCamera()
        setupView()
        setupPipeline()
        setupSamplerState()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        captureSession.startRunning()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession.stopRunning()
    }

    private func setupMetal() {
        device = MTLCreateSystemDefaultDevice()
        guard device != nil else {
            fatalError("Metal is not supported")
        }

        commandQueue = device.makeCommandQueue()
        uniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.stride,
                                        options: .storageModeShared)
    }

    private func setupML() {
        // Load YOLO or MobileNet model
        guard let modelURL = Bundle.main.url(forResource: "YOLOv3", withExtension: "mlmodelc"),
              let model = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL)) else {
            fatalError("Failed to load ML model")
        }

        visionModel = model
        detectionRequest = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            self?.processDetections(request.results as? [VNRecognizedObjectObservation] ?? [])
        }
        detectionRequest.imageCropAndScaleOption = .scaleFit
    }

    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            fatalError("Failed to setup camera")
        }

        videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]

        captureSession.addInput(input)
        captureSession.addOutput(videoOutput)

        DispatchQueue.global(qos: .background).async {
            self.captureSession.startRunning()
        }
    }

    private func processDetections(_ observations: [VNRecognizedObjectObservation]) {
        uniforms.numObjects = UInt32(min(observations.count, 20))

        for (index, observation) in observations.prefix(20).enumerated() {
            let bbox = observation.boundingBox
            uniforms.objectBoxes[index] = SIMD4<Float>(
                Float(bbox.origin.x),
                Float(bbox.origin.y),
                Float(bbox.size.width),
                Float(bbox.size.height)
            )
        }

        // Copy uniforms to the buffer
        guard let contents = uniformBuffer?.contents() else { return }
        memcpy(contents, &uniforms, MemoryLayout<Uniforms>.stride)
    }
    private func setupView() {
        metalView = MTKView(frame: view.bounds, device: device)
        metalView.delegate = self
        metalView.framebufferOnly = false
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.addSubview(metalView)

        // Set up vertex buffer
        vertexBuffer = device.makeBuffer(bytes: vertices,
                                       length: vertices.count * MemoryLayout<Float>.stride,
                                       options: [])
    }

    private func setupPipeline() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to create default library")
        }

        // Create pipeline descriptor
        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        // Configure vertex function
        guard let vertexFunction = library.makeFunction(name: "vertexShader") else {
            fatalError("Failed to create vertex function")
        }
        pipelineDescriptor.vertexFunction = vertexFunction

        // Configure fragment function
        guard let fragmentFunction = library.makeFunction(name: "fragmentShader") else {
            fatalError("Failed to create fragment function")
        }
        pipelineDescriptor.fragmentFunction = fragmentFunction

        // Configure color attachment
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat

        // Create pipeline state
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("Failed to create pipeline state: \(error)")
        }
    }

    private func setupTextureCache() {
           var cache: CVMetalTextureCache?
           CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
           textureCache = cache
       }

       private func createTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
           let width = CVPixelBufferGetWidth(pixelBuffer)
           let height = CVPixelBufferGetHeight(pixelBuffer)

           var textureRef: CVMetalTexture?
           let result = CVMetalTextureCacheCreateTextureFromImage(
               nil,
               textureCache!,
               pixelBuffer,
               nil,
               .bgra8Unorm,
               width,
               height,
               0,
               &textureRef
           )

           guard result == kCVReturnSuccess,
                 let texture = textureRef,
                 let metalTexture = CVMetalTextureGetTexture(texture) else {
               return nil
           }

           return metalTexture
       }

    private func setupSamplerState() {
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.sAddressMode = .clampToEdge
        samplerDescriptor.tAddressMode = .clampToEdge
        samplerState = device.makeSamplerState(descriptor: samplerDescriptor)
    }

}

// Add MTKViewDelegate conformance
extension MLMetalViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle resize if needed
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }

        // Set the render pipeline state
        renderEncoder.setRenderPipelineState(pipelineState)

        // Set the vertex buffer
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

        // Set the uniforms
        renderEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)

        // Set the texture and sampler state
        if let texture = texture {
            renderEncoder.setFragmentTexture(texture, index: 0)
            renderEncoder.setFragmentSamplerState(samplerState, index: 0)
        }

        // Draw the quad
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        // End encoding and commit
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension MLMetalViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                      didOutput sampleBuffer: CMSampleBuffer,
                      from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Create texture from camera frame
        if let newTexture = createTexture(from: pixelBuffer) {
            texture = newTexture
        }

        // Perform ML detection
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([detectionRequest])

        // Update Metal view for rendering
        DispatchQueue.main.async { [weak self] in
            self?.metalView.draw()
        }
    }
}

