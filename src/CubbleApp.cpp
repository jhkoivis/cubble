#include <iostream>
#include <cuda_profiler_api.h>

#include "CubbleApp.h"
#include "Fileio.h"
#include "Bubble.h"

using namespace cubble;

int CubbleApp::numSnapshots = 0;

CubbleApp::CubbleApp(const std::string &inF,
		     const std::string &saveF)
{
    env = std::make_shared<Env>(inF, saveF);
    env->readParameters();
    
    simulator = std::make_unique<Simulator>(env);
}

CubbleApp::~CubbleApp()
{
    saveSnapshotToFile();
    env->writeParameters();
}

void CubbleApp::run()
{
    cudaProfilerStart();
    std::cout << "**Starting the simulation setup.**\n" << std::endl;
    simulator->setupSimulation();

    std::cout << "Before profiler stop." << std::endl;
    cudaProfilerStop();
    std::cout << "After profiler stop" << std::endl;

    int numSteps = 0;
    const double phiTarget = env->getPhiTarget();
    double bubbleVolume = simulator->getVolumeOfBubbles();
    double phi = bubbleVolume / env->getSimulationBoxVolume();
    
    auto printPhi = [](double phi, double phiTarget) -> void
	{
	    std::cout << "Volume ratios: current: " << phi
	    << ", target: " << phiTarget
	    << std::endl;
	};

    std::cout << "Before phi & snapshot" << std::endl;
    printPhi(phi, phiTarget);
    std::cout << "After phi" << std::endl;
    saveSnapshotToFile();
    std::cout << "after snap" << std::endl;

    std::cout << "Starting the scaling of the simulation box." << std::endl;
    const bool shouldShrink = phi < phiTarget;
    const double scaleAmount = env->getScaleAmount() * (shouldShrink ? 1 : -1);
    while ((shouldShrink && phi < phiTarget) || (!shouldShrink && phi > phiTarget))
    {
	if (numSteps == 0)
	    cudaProfilerStart();
	else if (numSteps == 1)
	    cudaProfilerStop();
	
	env->setTfr(env->getTfr() - scaleAmount);
	simulator->integrate();
	phi = bubbleVolume / env->getSimulationBoxVolume();
	
	if (numSteps % 1000 == 0)
	    printPhi(phi, phiTarget);
	
	++numSteps;
    }
    
    std::cout << "Scaling took total of " << numSteps << " steps." << std::endl;
    printPhi(phi, phiTarget);
    saveSnapshotToFile();
    
    std::cout << "Starting the relaxation of the foam..." << std::endl;
    numSteps = 0;
    const int failsafe = 500;
    while (true)
    {
	double energy1 = simulator->getElasticEnergy();
	double time = 0;

	for (int i = 0; i < env->getNumStepsToRelax(); ++i)
	{
	    if (numSteps == 0 && i == 0)
		cudaProfilerStart();
	    else if (numSteps == 0 && i == 1)
		cudaProfilerStop();
	    
	    simulator->integrate(false);
	    time += env->getTimeStep();
	}
	
	double energy2 = simulator->getElasticEnergy();
	double deltaEnergy = energy1 == 0 ? 0
	    : std::abs(energy2 - energy1) / (energy1 * time);

	if (deltaEnergy < env->getMaxDeltaEnergy())
	{
	    std::cout << "Final delta energy " << deltaEnergy
		      << " after " << numSteps * env->getNumStepsToRelax()
		      << " steps."
		      << std::endl;
	    break;
	}
	else if (numSteps > failsafe)
	{
	    std::cout << "Over " << failsafe
		      << " steps taken and required delta energy not reached."
		      << " Check parameters."
		      << std::endl;
	    break;
	}
	else
	    std::cout << "Number of simulation steps relaxed: "
		      << numSteps * env->getNumStepsToRelax()
		      << ", delta energy: " << deltaEnergy
		      << std::endl;

	++numSteps;
    }

    saveSnapshotToFile();

    std::cout << "**Setup done.**"
	      <<"\n\n**Starting the simulation proper.**"
	      << std::endl;
    
    for (int i = 0; i < env->getNumIntegrationSteps(); ++i)
    {
	simulator->integrate(true);
	
	if (i % 1000 == 0)
	{
	    std::cout << "Current average radius after "
		      << i << " steps: "
		      << simulator->getAverageRadius()
		      << std::endl;
	}

	if (i % 100000 == 0)
	    saveSnapshotToFile();
    }

    saveSnapshotToFile();
    
    std::cout << "**Simulation has been finished.**\nGoodbye!" << std::endl;
}

void CubbleApp::saveSnapshotToFile()
{
    std::cout << "Writing a snap shot to a file..." << std::flush;

    std::vector<Bubble> tempVec;
    simulator->getBubbles(tempVec);
    
    std::stringstream ss;
    ss << env->getDataPath()
       << env->getSnapshotFilename()
       << numSnapshots
       << ".dat";

    std::string filename(ss.str());
    ss.clear();
    ss.str("");

    // Add descriptions here, when adding new things to the 'header' of the data file
    ss << "#--------------------------------------------------"
       << "\n# Lines starting with '#' are comment lines"
       << "\n#"
       << "\n# Format of data:"
       << "\n# left bottom back"
       << "\n# top front right"
       << "\n#"
       << "\n# bubble data: normalized position (x, y, z), unnormalized radius"
       << "\n#--------------------------------------------------";

    // Add the new things here.
    ss << "\n" << env->getLbb()
       << "\n" << env->getTfr();
    
    for (const auto &bubble : tempVec)
	ss << "\n" << bubble;
    
    fileio::writeStringToFile(filename, ss.str());
    ++numSnapshots;

    std::cout << " Done." << std::endl;
}
