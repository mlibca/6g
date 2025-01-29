#include <iostream>
#include <cuda_runtime.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <sys/socket.h>
#include <arpa/inet.h>


#include <thrust/reduce.h>



#define PACKET_SIZE 1500
#define NUM_PACKETS 1024*10000

__global__ void processPackets(unsigned char* packets, int numPackets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPackets) {
        struct iphdr* ipHeader = (struct iphdr*)(packets + idx * PACKET_SIZE);
        struct udphdr* udpHeader = (struct udphdr*)(packets + idx * PACKET_SIZE + sizeof(struct iphdr));

        // Example processing: Swap source and destination IP addresses
        unsigned int temp = ipHeader->saddr;
        ipHeader->saddr = ipHeader->daddr;
        ipHeader->daddr = temp;

        // Example processing: Swap source and destination ports
        unsigned short tempPort = udpHeader->source;
        udpHeader->source = udpHeader->dest;
        udpHeader->dest = tempPort;
		
		if( idx %100 == 0) 
		   printf(" idx : %u | ", idx ); 
    }
}

int main() {
    unsigned char* h_packets = (unsigned char*)malloc(NUM_PACKETS * PACKET_SIZE);
    unsigned char* d_packets;

    // Initialize packets with dummy data (in a real application, you'd receive these from a network interface)
    for (int i = 0; i < NUM_PACKETS * PACKET_SIZE; ++i) {
        h_packets[i] = static_cast<unsigned char>(i % 256);
    }

    cudaMalloc((void**)&d_packets, NUM_PACKETS * PACKET_SIZE);
    cudaMemcpy(d_packets, h_packets, NUM_PACKETS * PACKET_SIZE, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_PACKETS + threadsPerBlock - 1) / threadsPerBlock;
    processPackets<<<blocksPerGrid, threadsPerBlock>>>(d_packets, NUM_PACKETS);

    cudaMemcpy(h_packets, d_packets, NUM_PACKETS * PACKET_SIZE, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_packets);
    free(h_packets);

    std::cout << "Packet processing completed." << std::endl;

    return 0;
}
