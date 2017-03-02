package tuandn.test;

import java.util.Arrays;
import java.util.List;

import tuandn.SentenceTuple;

public class TestCases {

	public static class TestCase {
		public String sentence;
		public List<SentenceTuple> tuples;
		public TestCase(String sentence, List<SentenceTuple> tuples) {
			super();
			this.sentence = sentence;
			this.tuples = tuples;
		}
	}
}
