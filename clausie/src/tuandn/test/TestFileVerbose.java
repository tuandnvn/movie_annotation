package tuandn.test;

import java.io.IOException;
import java.util.List;

import de.mpii.clausie.ClausIE;
import tuandn.FileUtil;
import tuandn.ProcessSentence;
import tuandn.SentenceTuple;

public class TestFileVerbose {

	public static void main(String[] args) {
		ClausIE clausIE = new ClausIE();
		clausIE.initParser();

		ProcessSentence ps = new ProcessSentence(clausIE);

		String path = "../LSMDC task/LSMDC16_annos_val.desc.short.prob.txt";

		try {
			String content = FileUtil.readFile(path);

			String[] lines = content.split("\n");

			for (String line : lines) {
				String[] parts = line.split("\t");
				if (parts.length == 2) {
					int sentencePos = Integer.parseInt(parts[0]);

					String sentence = parts[1];
					try {
						List<SentenceTuple> sentenceTuples = ps.process(sentence, true, true);
						
						for (SentenceTuple sentenceTuple : sentenceTuples) {
							System.out.println(sentenceTuple.toBeautifulString());
						}
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
