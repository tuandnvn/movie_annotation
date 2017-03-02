package tuandn.test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import de.mpii.clausie.ClausIE;
import tuandn.FileUtil;
import tuandn.ProcessSentence;
import tuandn.SentenceTuple;

public class ProduceResults {
	public static void main(String[] args) {
		String valPath = "../LSMDC task/LSMDC16_annos_training.desc.csv";

		String valPathOutput = "../LSMDC task/LSMDC16_annos_training.desc.processed.txt";

		ClausIE clausIE = new ClausIE();
		clausIE.initParser();

		ProcessSentence ps = new ProcessSentence(clausIE);

		String valProblematicPath = "../LSMDC task/LSMDC16_annos_training.desc.prob.txt";
		try {
			String content = FileUtil.readFile(valPath);

			String[] lines = content.split("\n");

			// Number of input sentence that produce some output
			int produceCounter = 0;
			// Number of input sentence that produce exception
			int exceptionCounter = 0;
			// Number of clause forms in total
			int clauseCounter = 0;

			try (FileWriter fw = new FileWriter(new File(valPathOutput));
					FileWriter fw2 = new FileWriter(new File(valProblematicPath));) {
				for (String line : lines) {
					String[] parts = line.split("\t");

					if (parts.length == 2) {
						int sentencePos = Integer.parseInt(parts[0]);

						String sentence = parts[1];

						try {
							List<SentenceTuple> sentenceTuples = ps.process(sentence, false, true);

							for (SentenceTuple sentenceTuple : sentenceTuples) {
								fw.write("" + sentencePos);
								fw.write("\t");
								fw.write(sentenceTuple.toBeautifulString());
								fw.write("\n");
							}

							produceCounter += sentenceTuples.size() != 0 ? 1 : 0;
							clauseCounter += sentenceTuples.size();

							if (sentenceTuples.size() == 0) {
								System.out.println("CANNOT PROCESS");
								System.out.println(sentence);
								fw2.write("" + sentencePos);
								fw2.write("\t");
								fw2.write(sentence);
								fw2.write("\n");
							}

						} catch (Exception e) {
							fw.write("" + sentencePos);
							fw.write("\t");
							fw.write("EXCEPTION");
							fw.write("\n");
							exceptionCounter += 1;

							System.out.println("CANNOT PROCESS");
							System.out.println(sentence);
							
							fw2.write("" + sentencePos);
							fw2.write("\t");
							fw2.write(sentence);
							
							e.printStackTrace();
						}
					}
				}
			}

			System.out.println(String.format("Number of inputs = %d", lines.length));
			System.out.println(String.format("Number of input that produce some output = %d", produceCounter));
			System.out.println(String.format("Number of clause form produced = %d", clauseCounter));
			System.out.println(String.format("Number of input that produce Exception = %d", exceptionCounter));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
