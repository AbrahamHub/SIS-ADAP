from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.agent import Agent


class RoboCupAgent(Agent):
    class MyBehaviour(CyclicBehaviour):
        async def run(self):
            print("Ejecutando comportamiento cíclico...")

    async def setup(self):
        print("Agente RoboCup iniciado")
        b = self.MyBehaviour()
        self.add_behaviour(b)

class MainContainer:
    def __init__(self):
        self.agent = RoboCupAgent("RoboCupAgent@localhost", "secret")
    
    async def start(self):
        await self.agent.start()

if __name__ == "__main__":
    mc = MainContainer()
    mc.start()
