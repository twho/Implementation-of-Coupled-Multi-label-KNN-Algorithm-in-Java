package Classifiers;

import java.io.IOException;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.rmi.CORBA.Util;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MultiLabelKNN;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;

public class CMLkNN extends MultiLabelKNN {

	protected double smooth;
	private double[] PriorProbabilities;
	private double[] PriorNProbabilities;
	private double[][] CondProbabilities;
	private double[][] CondNProbabilities;
	// Similarity measurements
	private Double[] RF;
	private Double[][] freq;
	private double[] numoflabelsofins;
	int[] temp_Ci;
	int[][] newclsArray;
	int yell = 0;

	public CMLkNN(int numOfNeighbors, double smooth) {
		super(numOfNeighbors);
		this.smooth = smooth;
	}

	public CMLkNN() {
		super();
		this.smooth = 1.0;
	}

	public String globalInfo() {
		return "Class implementing the CML-kNN (Coupled Multi-Label k Nearest Neighbours) algorithm." + "\n\n"
				+ "For more information, see\n\n" + getTechnicalInformation().toString();
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		return null;
	}

	@Override
	protected void buildInternal(MultiLabelInstances train) throws Exception {
		super.buildInternal(train);
		PriorProbabilities = new double[numLabels];
		PriorNProbabilities = new double[numLabels];
		CondProbabilities = new double[numLabels][numOfNeighbors + 1];
		CondNProbabilities = new double[numLabels][numOfNeighbors + 1];
		RF = new Double[numLabels + 1]; // RF denotes occurance
		freq = new Double[numLabels + 1][numLabels + 1]; // co-occurance
		temp_Ci = new int[numLabels + 1];
		numoflabelsofins = new double[train.getNumInstances()];// 每個instance
		// System.out.println(temp_Ci);
		computeRF(temp_Ci);
		Newcls();
		ComputePrior();
		countnumoflabelsofins();
		// makePredictionInternal(null);

		if (getDebug()) {
			System.out.println("Computed Prior Probabilities");
			for (int i = 0; i < numLabels; i++) {
				System.out.println("Label " + (i + 1) + ": " + PriorProbabilities[i]);
			}
			System.out.println("Computed Posterior Probabilities");
			for (int i = 0; i < numLabels; i++) {
				System.out.println("Label " + (i + 1));
				System.out.println(i + " neighbours: " + CondProbabilities[i]);
				System.out.println(i + " neighbours: " + CondNProbabilities[i]);
			}
		}
	}

	private void ComputePrior() throws IOException {
		for (int i = 0; i < numLabels; i++) {
			int temp_Ci = 0;
			for (int j = 0; j < train.numInstances(); j++) {
				for (int k = 0; k < numOfNeighbors; k++) {
					if ((k + j) < train.numInstances()) {
						double value = newclsArray[j + k][i];
						if (value == 1) {
							temp_Ci++;
						}
					} else if ((k + j) >= train.numInstances()) {
						double value = newclsArray[j - k][i];
						if (value == 1) {
							temp_Ci++;
						}
					}
					// double value = Double.parseDouble(
					// train.attribute(labelIndices[i]).value((int)
					// train.instance(j).value(labelIndices[i])));
					// if (Utils.eq(value, 1.0)) {
					// temp_Ci++;
					// }
					// System.out.println(newclsArray[j][i]);

				}
			}
			// temp_Ci=9*temp_Ci;
			PriorProbabilities[i] = (smooth + temp_Ci) / (smooth * 2 + train.numInstances() * numLabels);
			PriorNProbabilities[i] = 1 - PriorProbabilities[i];
		}
	}

	private void countnumoflabelsofins() throws Exception {
		// for (int j = 0; j < train.numInstances(); j++) {
		// // System.out.println(numLabels);
		// Instances knn = new
		// Instances(lnn.kNearestNeighbours(train.instance(j), numOfNeighbors));
		// for (int i = 0; i < numLabels; i++) {
		//
		// double value = Double.parseDouble(
		// train.attribute(labelIndices[i]).value((int)
		// train.instance(j).value(labelIndices[i])));
		// if (value == 1) {
		// numoflabelsofins[j]++;
		// }
		// }
		// // System.out.println(numLabels);
		// }
		// for (int i = 0; i < numLabels; i++) {
		// PriorProbabilities[i] = ((smooth + clsSum[i])) / (smooth * 2 +
		// train.numInstances());
		// PriorNProbabilities[i] = 1 - PriorProbabilities[i];
		// }
	}

	private void computeRF(int[] temp_Ci) {
		for (int i = 0; i < numLabels; i++) {
			for (int j = 0; j < train.numInstances(); j++) {
				double value = Double.parseDouble(
						train.attribute(labelIndices[i]).value((int) train.instance(j).value(labelIndices[i])));
				// System.out.println(value);
				if (Utils.eq(value, 1.0)) {
					temp_Ci[i]++;
				}
			}
			// System.out.println(temp_Ci[i]);
			// RF[i] = ((double) temp_Ci[i] / (double) train.numInstances());
		}
	}

	private void Newcls() throws Exception {
		Double[][] neighborCls = new Double[numLabels][numLabels];
		// computeFreq(neighborCls);
		// countnumoflabelsofins();
		// double[] freqArray = new double[numLabels * numOfNeighbors+1];
		// double[] freqArrayN = new double[numLabels * numOfNeighbors+1];

		newclsArray = new int[train.numInstances()][numLabels];
		// double[][] newfreqArrayN = new
		// double[numLabels][train.numInstances()];
		double cooccurance = 0;
		for (int i = 0; i < numLabels; i++) {
			for (int k = 0; k < numLabels; k++) {
				double temp_times = 0;
				for (int j = 0; j < train.numInstances(); j++) {
					double valueI = Double.parseDouble(
							train.attribute(labelIndices[i]).value((int) train.instance(j).value(labelIndices[i])));
					double valueK = Double.parseDouble(
							train.attribute(labelIndices[k]).value((int) train.instance(j).value(labelIndices[k])));
					if (valueI * valueK == 1) {
						temp_times++;
					}
				}
				// System.out.println(temp_Ci[k]);
				cooccurance = temp_times / ((double) Math.max(temp_Ci[i], temp_Ci[k]));
				neighborCls[i][k] = cooccurance;
				neighborCls[k][i] = cooccurance;
				// System.out.println(neighborCls[i][k]);
			}
		}
		for (int j = 0; j < train.numInstances(); j++) {
			// System.out.println(train.numInstances());
			Instances knn = new Instances(lnn.kNearestNeighbours(train.instance(j), numOfNeighbors));
			for (int i = 0; i < numLabels; i++) {

				double value = Double.parseDouble(
						train.attribute(labelIndices[i]).value((int) train.instance(j).value(labelIndices[i])));
				// System.out.println(value);
				if (value == 1) {
					numoflabelsofins[j]++;
				}
			}
			// System.out.println(numoflabelsofins[j]);
		}
		int a = 0;
		int b = 0;
		for (int j = 0; j < train.numInstances(); j++) {
			// System.out.println(numoflabelsofins[j]);
			Instances knn = new Instances(lnn.kNearestNeighbours(train.instance(j), numOfNeighbors + 1));
			for (int i = 0; i < numLabels; i++) {
				double x = 0;
				for (int q = 0; q < numLabels; q++) {
					double value = Double.parseDouble(
							train.attribute(labelIndices[q]).value((int) train.instance(j).value(labelIndices[q])));
					x += value * neighborCls[i][q];
					// System.out.println(neighborCls[i][q]);
				}
				// System.out.println(x);
				// System.out.println(numoflabelsofins[j]);
				double y = x / (numoflabelsofins[j]);
				// System.out.println(y);
				if (y > 0.53) {
					newclsArray[j][i] = 1;
					a++;
				} else {
					newclsArray[j][i] = 0;
					b++;
				}
				// System.out.println(newclsArray[j][i]);
			}
		}
		// System.out.println(a);
		// }
		//////////////////////////////////////////////////////////////////////////////
		// private void ComputeCond() throws Exception {
		// Newcls();
		int[][] temp_Ci = new int[numLabels][numOfNeighbors + 1];
		int[][] temp_NCi = new int[numLabels][numOfNeighbors + 1];

		for (int j = 0; j < train.numInstances(); j++) {
			// Instances knn = new
			// Instances(lnn.kNearestNeighbours(train.instance(j),
			// numOfNeighbors));
			// System.out.println(knn);
			// now compute values of temp_Ci and temp_NCi for every class label
			for (int i = 0; i < numLabels; i++) {
				int aces = 0; // num of aces in Knn for j
				for (int k = 0; k < numOfNeighbors; k++) {
					if ((j + k) < train.numInstances()) {
						double value = newclsArray[j + k][i];
						if (value == 1) {
							aces++;
						}
					} else if ((j + k) >= train.numInstances()) {
						double value = newclsArray[j - k][i];
						if (value == 1) {
							aces++;
						}
					}

					// double value = Double.parseDouble(
					// train.attribute(labelIndices[j]).value((int)
					// knn.instance(k).value(labelIndices[j])));
				}
				// if (aces > 9) {
				// System.out.println(aces);
				// } // raise the counter of temp_Ci[j][aces] and
				// temp_NCi[j][aces]
				// by 1
				// label j與該instance i的關係
				if (newclsArray[j][i] == 1) {
					temp_Ci[i][aces]++;
				} else {
					temp_NCi[i][aces]++;
				}
				// System.out.println(temp_Ci[i][aces]);
			}
		}

		// compute CondProbabilities[i][..] for labels based on temp_Ci[]
		for (int i = 0; i < numLabels; i++) {
			int temp1 = 0;
			int temp2 = 0;
			for (int j = 0; j < numOfNeighbors + 1; j++) {
				temp1 += temp_Ci[i][j];
				temp2 += temp_NCi[i][j];
			}
			for (int j = 0; j < numOfNeighbors + 1; j++) {
				CondProbabilities[i][j] = (smooth + temp_Ci[i][j])
						/ (smooth * (numLabels * numOfNeighbors + 1) + temp1);
				CondNProbabilities[i][j] = (smooth + temp_NCi[i][j])
						/ (smooth * (numLabels * numOfNeighbors + 1) + temp2);
			}
		}
		//////////////////////////////////////////
	}

	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
		double[] confidences = new double[numLabels];
		boolean[] predictions = new boolean[numLabels];
		// 問題：此處的knn是怎麼取得？try指令？
		Instances knn = null;
		try {
			knn = new Instances(lnn.kNearestNeighbours(instance, numOfNeighbors));
		} catch (Exception ex) {
			Logger.getLogger(MLkNN.class.getName()).log(Level.SEVERE, null, ex);
		}
		// System.out.println(numLabels);
		yell++;
		// System.out.println(yell);
		for (int i = 0; i < numLabels; i++) {
			// compute sum of aces in KNN
			int aces = 0; // num of aces in Knn for i
			for (int k = 0; k < numOfNeighbors; k++) {
				// if ((yell + k) < train.numInstances()) {
				// double value = newclsArray[yell+k][i];
				// if (value == 1) {
				// aces++;
				// }
				// else if ((yell + k) >= train.numInstances()) {
				// value = newclsArray[yell-k][i];
				// if (value == 1) {
				// aces++;
				// }
				// }
				// }

				double value = Double.parseDouble(
						train.attribute(labelIndices[i]).value((int) knn.instance(k).value(labelIndices[i])));
				if (Utils.eq(value, 1.0)) {
					aces++;
				}
				// j++;
			}
			// if(aces>9){
			// j=j+numOfNeighbors;
			double Prob_in = PriorProbabilities[i] * CondProbabilities[i][aces];
			double Prob_out = PriorNProbabilities[i] * CondNProbabilities[i][aces];
			if (Prob_in > Prob_out) {
				predictions[i] = true;
			} else if (Prob_in < Prob_out) {
				predictions[i] = false;
			} else {
				Random rnd = new Random();
				predictions[i] = (rnd.nextInt(2) == 1) ? true : false;
			}
			// ranking function
			confidences[i] = Prob_in / (Prob_in + Prob_out);
		}
		MultiLabelOutput mlo = new MultiLabelOutput(predictions, confidences);
		return mlo;
	}
}