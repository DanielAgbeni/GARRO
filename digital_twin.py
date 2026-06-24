import math

class DigitalTwinMM1K:
    def __init__(self, capacity_K):
        """
        Initializes the digital twin switch interface.
        K: Maximum number of packets the switch buffer can hold.
        """
        self.K = capacity_K

    def evaluate_traffic_state(self, arrival_rate_lambda, service_rate_mu):
        """
        Calculates queue metrics based on M/M/1/K queueing theory.
        lambda: Packet arrival rate (packets per second)
        mu: Packet processing/transmission rate (packets per second)
        """
        # Traffic intensity (rho)
        rho = arrival_rate_lambda / service_rate_mu
        
        # Calculate P_0: The probability the queue is completely empty
        if rho == 1:
            P_0 = 1 / (self.K + 1)
        else:
            P_0 = (1 - rho) / (1 - math.pow(rho, self.K + 1))
            
        # Calculate P_K: The probability of buffer overflow (Packet Loss)
        if rho == 1:
            P_overflow = 1 / (self.K + 1)
        else:
            P_overflow = P_0 * math.pow(rho, self.K)
            
        # Calculate Expected Queue Length (E[Q])
        if rho == 1:
            expected_queue = self.K / 2
        else:
            numerator = rho * (1 - (self.K + 1) * math.pow(rho, self.K) + self.K * math.pow(rho, self.K + 1))
            denominator = (1 - rho) * (1 - math.pow(rho, self.K + 1))
            expected_queue = numerator / denominator
            
        # Little's Law: Calculate average latency (W) for packets that successfully enter
        effective_lambda = arrival_rate_lambda * (1 - P_overflow)
        if effective_lambda > 0:
            latency = expected_queue / effective_lambda
        else:
            latency = 0
            
        return {
            "traffic_intensity": round(rho, 4),
            "overflow_probability_loss": round(P_overflow, 4),
            "expected_queue_depth": round(expected_queue, 2),
            "estimated_latency_sec": round(latency, 6)
        }

# --- Test the Simulator ---
if __name__ == "__main__":
    # Simulate a switch with a buffer capacity of 100 packets
    simulator = DigitalTwinMM1K(capacity_K=100)
    
    print("--- Scenario 1: Normal Traffic ---")
    # 800 packets/sec arriving, switch can process 1000 packets/sec
    result_normal = simulator.evaluate_traffic_state(arrival_rate_lambda=800, service_rate_mu=1000)
    print(result_normal)
    
    print("\n--- Scenario 2: Severe Microburst (Elephant Flow) ---")
    # 1200 packets/sec arriving, switch can process 1000 packets/sec
    result_burst = simulator.evaluate_traffic_state(arrival_rate_lambda=1200, service_rate_mu=1000)
    print(result_burst)