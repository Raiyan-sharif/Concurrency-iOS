//
//  ViewController.swift
//  ConcurrencyiOS
//
//  Created by BJIT on 23/12/24.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        
        let label = "com.raywenderlich.mycoolapp.networking"
        let queue = DispatchQueue(label: label, attributes: .concurrent)

        // .userinteractive QoS is recommended for tasks that the user directly interacts with. UI-updating calculations, animations or anything needed to keep the UI responsive and fast
        let queue_userInteractive = DispatchQueue.global(qos: .userInteractive)

        //userInitiated queue should be used when the user kicks off a task from the UI that needs to happen immediately, but can be done asynchronously. For example, you may need to open a document or read from a local database
        let queue_userInitiated = DispatchQueue.global(qos: .userInitiated)

        //.utility dispatch queue for tasks that would typically include a progress indicator such as long-running computations, I/O, networking or continuous data feeds
        let queue_utility = DispatchQueue.global(qos: .utility)

        //For tasks that the user is not directly aware of you should use the .background queue. They don’t require user interaction and aren’t time sensitive. Prefetching, database maintenance, synchronizing remote servers and performing backups are all great examples.
        let queue_background = DispatchQueue.global(qos: .background)


    }


}

