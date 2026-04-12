from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Space exploration is the study and discovery of outer space through the use of advanced technology, spacecraft, and scientific research. It began in the mid-20th century when the first satellites were launched into orbit. In 1969, the first humans landed on the Moon, marking a major milestone in history and proving that space travel was possible.

Since then, space missions have expanded our understanding of the universe. Robotic spacecraft and rovers have explored planets like Mars, sending back valuable data about their surfaces and climates. Space telescopes have captured detailed images of distant stars and galaxies, helping scientists learn more about how the universe was formed.

Space exploration also benefits life on Earth. Satellites are used for communication, weather forecasting, navigation, and environmental monitoring. Many technologies developed for space missions have improved medicine, engineering, and computing.

Today, both governments and private companies are working toward new goals, including returning to the Moon and sending humans to Mars. Although space exploration is expensive and challenging, it continues to inspire innovation and curiosity, pushing humanity to explore beyond our planet and expand our knowledge of the cosmos."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=20,
    separators=""
)

result = splitter.split_text(text)

print(f"\nTotal number of chunks: {len(result)}\n")
print("="*80)
print(result)