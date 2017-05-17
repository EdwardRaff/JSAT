/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.clustering.kmeans;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.KClustererBase;
import jsat.distributions.Uniform;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class ElkanKMeansTestNew {
	static private SimpleDataSet easyData10;
	static private ExecutorService ex;
	/**
	 * Used as the starting seeds for k-means clustering to get consistent
	 * desired behavior
	 */
	static private List<Vec> seeds;

	public ElkanKMeansTestNew() {
	}

	@BeforeClass
	public static void setUpClass() throws Exception {
		GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15),
				new XORWOW(), 2, 5);
		easyData10 = gdg.generateData(110);
		readEasyData(Paths.get("./test-data/clustering/kMeans/easyData"));
		ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

	}

	private static void readEasyData(Path path) {
		try {
			BufferedReader br = Files.newBufferedReader(path);
			String line = "";
			List<DataPoint> data = new ArrayList<DataPoint>();
			while ((line = br.readLine()) != null) {
				String[] split = line.split(" ");
				String[] numbers = split[1].split(",");
				String option = split[4].replace(",", "");
				double[] doubles = new double[] {
						Double.parseDouble(numbers[0]),
						Double.parseDouble(numbers[1]) };
				DataPoint dp = new DataPoint(new DenseVector(doubles),
						new int[] { Integer.parseInt(option) - 1 },
						easyData10.getCategories());
				// System.out.println(dp);
				data.add(dp);
			}
			easyData10 = new SimpleDataSet(data);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	@AfterClass
	public static void tearDownClass() throws Exception {
		ex.shutdown();
	}

	@Before
	public void setUp() {
		seeds = parseSeeds(Paths.get("./test-data/clustering/kMeans/seeds"));
		readEasyData(Paths.get("./test-data/clustering/kMeans/easyData"));
	}

	/**
	 * Test of cluster method, of class ElkanKMeans.
	 */
	@Test
	public void testCluster_DataSet_int() {
		System.out.println("cluster(dataset, int)");
		ElkanKMeans kMeans = new ElkanKMeans(new EuclideanDistance());
		int[] assignment = new int[easyData10.getSampleSize()];
		kMeans.cluster(easyData10, null, 10, seeds, assignment, true, null,
				true);
		List<List<DataPoint>> clusters = KClustererBase
				.createClusterListFromAssignmentArray(assignment, easyData10);
		assertEquals(10, clusters.size());
		Set<Integer> seenBefore = new HashSet<Integer>();
		for (List<DataPoint> cluster : clusters) {
			int thisClass = cluster.get(0).getCategoricalValue(0);
			assertFalse(seenBefore.contains(thisClass));
			for (DataPoint dp : cluster)
				assertEquals(thisClass, dp.getCategoricalValue(0));
		}
	}

	/**
	 * Test of cluster method, of class ElkanKMeans.
	 */
	@Test
	public void testCluster_3args_2() {
		System.out.println("cluster(dataset, int, threadpool)");
		System.out.println("Seeds");

		for (Vec v : seeds)
			System.out.println(v);
		ElkanKMeans kMeans = new ElkanKMeans(new EuclideanDistance());
		int[] assignment = new int[easyData10.getSampleSize()];
		kMeans.cluster(easyData10, null, 10, seeds, assignment, true, ex, true);
		Vec[] trueMeans = new Vec[10];
		System.out.println("True");
		for (int i = 0; i < trueMeans.length; i++) {
			Vec v = new DenseVector(2);
			for (int j = 0; j < easyData10.getSampleSize(); j++) {
				DataPoint dp = easyData10.getDataPoint(j);
				if (dp.getCategoricalValue(0) == i)
					v.mutableAdd(dp.getNumericalValues());

			}
			v.mutableDivide(110);
			trueMeans[i] = v;
			System.out.println(v);
		}
		System.out.println("Found");
		for (Vec v : seeds)
			System.out.println(v);
		for (int i = 0; i < seeds.size(); i++) {
			for (int j = 0; j < 2; j++) {
				assertEquals(
						"seeds and true centroids are not the same for index="
								+ i, seeds.get(i).get(j), trueMeans[i].get(j),
						0.000000001);
			}
		}
		List<List<DataPoint>> clusters = KClustererBase
				.createClusterListFromAssignmentArray(assignment, easyData10);
		assertEquals(10, clusters.size());
		Set<Integer> seenBefore = new HashSet<Integer>();
		for (List<DataPoint> cluster : clusters) {
			int thisClass = cluster.get(0).getCategoricalValue(0);
			assertFalse(seenBefore.contains(thisClass));
			for (DataPoint dp : cluster)
				assertEquals(thisClass, dp.getCategoricalValue(0));
		}

	}

	private List<Vec> parseSeeds(Path path) {
		try {
			BufferedReader br = Files.newBufferedReader(path);
			String line = "";
			List<Vec> data = new ArrayList<Vec>();
			while ((line = br.readLine()) != null) {
				String[] split = line.split(",");
				double[] doubles = new double[] { Double.parseDouble(split[0]),
						Double.parseDouble(split[1]) };
				data.add(new DenseVector(doubles));
			}
			return data;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
}
