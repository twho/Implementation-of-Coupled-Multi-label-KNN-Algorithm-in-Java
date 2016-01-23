package Main;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import Classifiers.CMLkNN;
import Classifiers.MLkNN;
import DataSource.DataSource;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;

public class Main implements DataSource {

	public static void main(String[] args) throws Exception {

		String filePathArff = MultiLabelDataSet[2] + ARFF;
		String filePathXml = MultiLabelDataSet[2] + XML;

		MultiLabelInstances dataset = new MultiLabelInstances(filePathArff,
				filePathXml);

		CMLkNN learner1 = new CMLkNN(9, 1);
		Evaluator eval = new Evaluator();
		MultipleEvaluation results;

		int numFolds = 10;
		results = eval.crossValidate(learner1, dataset, numFolds);
		saveOutPut("CMLkNN", results.toString());
		System.out.println(results);
	}

	private static void saveOutPut(String learningMethod, String results) {
		String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss")
				.format(Calendar.getInstance().getTime());
		try {
			PrintStream out = new PrintStream(new FileOutputStream(
					learningMethod + "_" + timeStamp + "_output.txt"));
			out.println(results);
			out.close();
		} catch (IOException exception) {
			System.out.println("Error during file reading/writing");
		}
	}
}