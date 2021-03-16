/*
 * This code contributed under the public domain. 
 */
package jsat.classifiers.linear;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import jsat.DataSet;
import jsat.SimpleWeightVectorModel;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.WarmClassifier;
import jsat.classifiers.evaluation.ClassificationScore;
import jsat.classifiers.svm.DCDs;
import jsat.datatransform.ProjectionTransform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.lossfunctions.LogisticLoss;
import jsat.math.OnLineStatistics;
import jsat.parameters.GridSearch;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.RegressionModelEvaluation;
import jsat.regression.Regressor;
import jsat.regression.WarmRegressor;
import jsat.regression.evaluation.RegressionScore;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 * This class implements a general purpose method for endowing any linear
 * classification or regression model with the ability to train in parallel
 * using multiple cores. It does this by training an independent model for each
 * core, and using a final "aggregating" model to determine how to combine the
 * results from each core.<br>
 * <br>
 * See: ﻿Izbicki, M., & Shelton, C. R. (2020). Distributed Learning of
 * Non-convex Linear Models with One Round of Communication. In ECML-PKDD (pp.
 * 197–212).
 *
 * @author Edward Raff
 */
public class OWA implements Classifier, Regressor, Parameterized, SingleWeightVectorModel, WarmClassifier
{
    protected int min_points_per_core = 1000;
    protected int sample_multipler = 3;
    @ParameterHolder
    private SingleWeightVectorModel base_learner;
    private boolean estimate_cv_scores = false;
    private List<ClassificationScore> scores_c = new ArrayList<>();
    private List<RegressionScore> scores_r = new ArrayList<>();
    private List<OnLineStatistics> scores_stats = new ArrayList<>();
    private boolean warmTraining = false;
    private List<SimpleWeightVectorModel> prev_solutions = null;
    
    protected Vec w;
    protected double bias;

    /**
     * Constructs a new OWA model. 
     * 
     * @param base_learner the classifier or regressor to use as the base linear model, which will be parallelized by OWA. 
     */
    public OWA(SingleWeightVectorModel base_learner)
    {
	this.base_learner = base_learner;
    }

    /**
     * Copy constructor
     * @param toClone the object to copy
     */
    public OWA(OWA toClone)
    {
	this.min_points_per_core = toClone.min_points_per_core;
	this.sample_multipler = toClone.sample_multipler;
	this.base_learner = (SingleWeightVectorModel) toClone.base_learner.clone();
	this.estimate_cv_scores = toClone.estimate_cv_scores;
	if(toClone.w != null)
	{
	    this.w = toClone.w.clone();
	    this.bias = toClone.bias;
	}
	
	this.scores_c = toClone.scores_c.stream().map(s->s.clone()).collect(Collectors.toList());
	this.scores_r = toClone.scores_r.stream().map(s->s.clone()).collect(Collectors.toList());
	this.scores_stats = toClone.scores_stats.stream().map(s->s.clone()).collect(Collectors.toList());
	this.warmTraining = toClone.warmTraining;
	if(toClone.prev_solutions != null)
	{
	    this.prev_solutions = toClone.prev_solutions.stream().map(s->s.clone()).collect(Collectors.toList());
	}
	
    }

    /**
     * At a small incremental cost to training time, OWA can perform an
     * approximate cross-validation estimate of it's performance. K-fold cross
     * validation normally increases training time by a factor of K, but OWA can
     * estimate this for little additional work (often less than 10% additional
     * time). <br>
     * <br>
     * This is not done by default, but can be enabled with this
     * function. If enabled use {@link #addScore(jsat.classifiers.evaluation.ClassificationScore)
     * } or {@link #addScore(jsat.regression.evaluation.RegressionScore) } to
     * add scoring methods based on whether you are performing a classification
     * or regression problem respectively.
     *
     * @param estimate_cv_scores {@code true} to estimate CV performance while
     *                           training, or {@code false} to not perform the
     *                           additional work.
     */
    public void setEstimateCV(boolean estimate_cv_scores)
    {
	this.estimate_cv_scores = estimate_cv_scores;
    }

    /**
     *
     * @return {@code true} is OWA will attempt to estimate cross-validation
     *         results during training.
     */
    public boolean issetEstimateCV()
    {
	return estimate_cv_scores;
    }

    /**
     * The OWA algorithm can only perform warm-started training if the base
     * algorithm given also supports warm-training, and so is not enabled by
     * default. If the base model does support warm training, and you want to
     * use warm-starts of OWA models, setting this to {@code true} will cause
     * the OWA model to keep internal copies of the previous intermediate
     * solutions, which increases overall memory use. These are used to
     * warm-start the sub-models of future OWA training runs, and the OWA code
     * will keep track of which subset of data it used for each processor in
     * order to avoid any data leakage across subsets.
     *
     * @param warmTraining {@code true} if you plan to do warm stars with OWA,
     *                     or {@code false} by default to not.
     */
    public void setWarmTraining(boolean warmTraining)
    {
	this.warmTraining = warmTraining;
    }

    public boolean isWarmTraining()
    {
	return warmTraining;
    }
    
    
    
    /**
     * OWA can perform its own cross-validation estimates of performance. This
     * method adds a scoring method similar to
     * {@link ClassificationModelEvaluation#addScorer} to estimate performance.
     *
     * @param score 
     */
    public void addScore(ClassificationScore score)
    {
	scores_c.add(score);
    }

    /**
     * Gets the statistics associated with each score in one map. If no
     * estimated cross validation was performed, or no scores added, the result
     * will be an empty map. This method should be called only if OWA was trained on a classification problem. 
     *
     * @return the result statistics for each score as a map.
     */
    public Map<ClassificationScore, OnLineStatistics> getScoreStatsC()
    {
	Map<ClassificationScore, OnLineStatistics> results = new HashMap<>();
	for (int i = 0; i < Math.min(scores_c.size(), scores_stats.size()); i++)
	    results.put(scores_c.get(i), scores_stats.get(i));
	return results;
    }
    
    /**
     * Gets the statistics associated with each score in one map. If no
     * estimated cross validation was performed, or no scores added, the result
     * will be an empty map. This method should be called only if OWA was trained on a regression problem. 
     *
     * @return the result statistics for each score as a map.
     */
    public Map<RegressionScore, OnLineStatistics> getScoreStatsR()
    {
	Map<RegressionScore, OnLineStatistics> results = new HashMap<>();
	for (int i = 0; i < Math.min(scores_r.size(), scores_stats.size()); i++)
	    results.put(scores_r.get(i), scores_stats.get(i));
	return results;
    }
    
    /**
     * OWA can perform its own cross-validation estimates of performance. This
     * method adds a scoring method similar to
     * {@link RegressionModelEvaluation#addScorer} to estimate performance.
     * @param score 
     */
    public void addScore(RegressionScore score)
    {
	scores_r.add(score);
    }
    
    
    private void trainWork(final int requested_cores, DataSet dataSet, boolean parallel, Object warmSolution)
    {
	
	final int d = dataSet.getNumFeatures();
	final int N = dataSet.size();
	
	/**
	 * How many "machines" (read, cores) are we using
	 */
	final int m = requested_cores <= 0 ? Math.min(Math.min(SystemInfo.LogicalCores, dataSet.size()/min_points_per_core), d/2+1) : requested_cores;
//	System.out.println("Using " +  m + " cores");
	
	List<Object> warm_starts;
	if(warmTraining && warmSolution != null)
	{
	    if(!(base_learner instanceof WarmClassifier || base_learner instanceof WarmRegressor))
		throw new FailedToFitException("Base class " + base_learner.getClass().getSimpleName() + " can not be trained via warm starts");
	    warm_starts = new ArrayList<>();
	    //Is the warm start an OWA with a per-split solution?
	    if(warmSolution instanceof OWA && ((OWA)warmSolution).prev_solutions != null)
		for(SimpleWeightVectorModel sol : ((OWA)warmSolution).prev_solutions)
		    warm_starts.add(sol);
	    else//its not, just use global solution as generic start
	    {
		warm_starts.add(warmSolution);
	    }
	    while(warm_starts.size() < m)//padd out to the number of models we are going to train to simplify code
		warm_starts.add(warm_starts.get(warm_starts.size()-1));//all point to same obj so cheap
	}
	else
	    warm_starts = null;
	
	List<? extends DataSet<? extends DataSet>> splits = dataSet.cvSet(m, RandomUtil.getRandom(m*dataSet.size()));//Using a deterministic random seed so that if the models use warm-starts we can get the warm-starts to re-use the same sub-splits 
	
//	System.out.println("Training local models");
	List<SimpleWeightVectorModel> erms = ParallelUtils.streamP(IntStream.range(0, splits.size()), parallel).mapToObj(i->
	{
	    DataSet<? extends DataSet> data = splits.get(i);
//	    System.out.println("Training on " + data.size() + "/" + dataSet.size() + " local samples");
	    SimpleWeightVectorModel w_i = base_learner.clone();
	    Object warm_w_i = warm_starts == null ? null : warm_starts.get(i);
	    if(w_i instanceof Classifier)
	    {
		if(w_i instanceof WarmClassifier && warm_w_i != null)
		    ((WarmClassifier)w_i).train((ClassificationDataSet)data, ((Classifier)warm_w_i), false);
		else
		    ((Classifier)w_i).train((ClassificationDataSet) data);
	    }
	    else
	    {
		if(w_i instanceof WarmClassifier && warm_w_i != null)
		    ((WarmRegressor)w_i).train((RegressionDataSet)data, ((Regressor)warm_w_i), false);
		else
		    ((Regressor)w_i).train((RegressionDataSet) data);
	    }
	    return w_i;
	}).collect(Collectors.toList());
	
	if(warmTraining)
	    this.prev_solutions = erms;
	
	
//	System.out.println("Sample & Project");
	//Lets build out projection matrix & transform 
	Matrix W = new DenseMatrix(m, d);
	Vec b = new DenseVector(m);
	for(int i = 0; i < m; i++)
	{
	    SingleWeightVectorModel w_i = (SingleWeightVectorModel) erms.get(i);
	    w_i.getRawWeight().copyToRow(W, i);
	    b.set(i, w_i.getBias());
	}
	ProjectionTransform t = new ProjectionTransform(W, b);
	
	
	//lets prepare the smaller sub-set of data Z that is used in round 2.
	double sub_sample_frac = Math.min(Math.max(sample_multipler*m/(double)d+40/(double)N, (m+40)*sample_multipler/(double)(N/m)), 1.0);

	//Sub-sample each split to get all the parts of Z_owa. Done this way so that we can do an easy CV estimate if desired.
	//TODO try replacing this with some kind of coreset selection based on trained models
	List<? extends DataSet<? extends DataSet>> Z_owa_splits = splits.parallelStream()
	.map(data ->
	{
	    DataSet z_i = data.randomSplit(sub_sample_frac).get(0);
//	    System.out.println("CV Estimate chunk contribution " + z_i.size() + " based on frac " + sub_sample_frac);


	    if(!data.rowMajor())//orig col-major may have been OK, but we want row-major now
	    {
		Iterator<DataPoint> orig_iter = data.getDataPointIterator();
		int pos = 0;

		if(data instanceof ClassificationDataSet)
		{
		    ClassificationDataSet new_data = new ClassificationDataSet(W.rows(), new CategoricalData[0], ((ClassificationDataSet)z_i).getPredicting());
		    while(orig_iter.hasNext())
			new_data.addDataPoint(t.transform(orig_iter.next()), ((ClassificationDataSet)z_i).getDataPointCategory(pos++));
		    z_i = new_data;
		}
		else
		{
		    RegressionDataSet new_data = new RegressionDataSet(W.rows(), new CategoricalData[0]);
		    while(orig_iter.hasNext())
			new_data.addDataPoint(t.transform(orig_iter.next()), ((RegressionDataSet)z_i).getTargetValue(pos++));
		    z_i = new_data;
		}
	    }
	    else//apply transform easy-peasy
		z_i.applyTransform(t);
	    return (DataSet<? extends DataSet>) z_i;
	})
	.collect(Collectors.toList());
	
	
	
	if(estimate_cv_scores)
	{
//	    System.out.println("CV Estimate Steps");
	    scores_stats.clear();
	    ParallelUtils.streamP(IntStream.range(0, m), true).forEach(id->
	    {
		//build a dataset Z_{-i} to have all the results from the other corpra, but this one
		SimpleWeightVectorModel Z_model;
		DataSet Z_owa_mi;
		if(dataSet instanceof ClassificationDataSet)
		{
		    Z_owa_mi = ClassificationDataSet.comineAllBut((List<ClassificationDataSet>)Z_owa_splits, id);
		    LogisticRegressionDCD lr = new LogisticRegressionDCD();
		    lr.setUseBias(false);
		    Z_model = lr;
		}
		else
		{
		    Z_owa_mi = RegressionDataSet.comineAllBut((List<RegressionDataSet>)Z_owa_splits, id);
		    DCDs dcd = new DCDs();
		    dcd.setUseBias(false);
		    Z_model = dcd;
		}
		//We need to remove the columns associated with model i's prediction. Lazy option, lets apply a transform that simply zeros out these values so that the index match up still
		Z_owa_mi.applyTransform(dp ->
		{
		    Vec v = dp.getNumericalValues().clone();
		    v.set(id, 0);
		    return new DataPoint(v);
		});
//		System.out.println("\tUsing " + Z_owa_mi.size() + " samples to estimate mixing ratio");
		//train a model on Z_{-i}
		SimpleWeightVectorModel cv_model;
		GridSearch rs = new GridSearch((Classifier)Z_model, 5); //lazy but OK b/c DCD is also a classifier.
		rs.setUseWarmStarts(true); //since each processor is doing it's own search sequentially, use warm-starts to speed up as much as we can
		rs.autoAddParameters(Z_owa_mi, 9);
		rs.setTrainModelsInParallel(false);
		rs.setTrainFinalModel(true);
		if(Z_owa_mi instanceof ClassificationDataSet)
		{
		    rs.train((ClassificationDataSet) Z_owa_mi, false);
		    cv_model = (SimpleWeightVectorModel) rs.getTrainedClassifier();
		}
		else
		{
		    rs.train((RegressionDataSet) Z_owa_mi, false);
		    cv_model = (SimpleWeightVectorModel) rs.getTrainedRegressor();
		}
		//make super sure we don't use anything from the current id, its held out! 
		cv_model.getRawWeight(0).set(id, 0);
		
		Vec w_mi = new DenseVector(d);
		Vec b_mi = new DenseVector(1);
		accumulateUpdates(m, cv_model, w_mi, b_mi, W, b);

		
		if(Z_owa_mi instanceof ClassificationDataSet)
		{
		    ClassificationDataSet cds = (ClassificationDataSet) splits.get(id);
		    List<ClassificationScore> scores = scores_c.stream().map(s->s.clone()).collect(Collectors.toList());
		    for(ClassificationScore s : scores)
			s.prepare(cds.getPredicting());
		    int pos = 0;
		    Iterator<DataPoint> iter = cds.getDataPointIterator();
		    Vec weights = cds.getDataWeights();
		    while(iter.hasNext())
		    {
			CategoricalResults result = LogisticLoss.classify(w_mi.dot(iter.next().getNumericalValues()) + b_mi.get(0));
			for(ClassificationScore s : scores)
			    s.addResult(result, cds.getDataPointCategory(pos), weights.get(pos));
			pos++;
		    }
		    
		    //record the result
		    synchronized(scores_stats)
		    {
			if(scores_stats.isEmpty())
			    for(ClassificationScore s : scores)
				scores_stats.add(new OnLineStatistics());
			for(int i = 0; i < scores.size(); i++)
			    scores_stats.get(i).add(scores.get(i).getScore());
		    }
		}
		else//regression case
		{
		    RegressionDataSet rds = (RegressionDataSet) splits.get(id);
		    List<RegressionScore> scores = scores_r.stream().map(s->s.clone()).collect(Collectors.toList());

		    int pos = 0;
		    Iterator<DataPoint> iter = rds.getDataPointIterator();
		    Vec weights = rds.getDataWeights();
		    while(iter.hasNext())
		    {
			double result = w_mi.dot(iter.next().getNumericalValues()) + b_mi.get(0);
			for(RegressionScore s : scores)
			    s.addResult(result, rds.getTargetValue(pos), weights.get(pos));
			pos++;
		    }
		    //record the result
		    synchronized(scores_stats)
		    {
			if(scores_stats.isEmpty())
			    for(RegressionScore s : scores)
				scores_stats.add(new OnLineStatistics());
			for(int i = 0; i < scores.size(); i++)
			    scores_stats.get(i).add(scores.get(i).getScore());
		    }
		}
	    });
	}


//	System.out.println("Train Avg");
	SimpleWeightVectorModel Z_model;
	DataSet Z_owa;
	if(dataSet instanceof ClassificationDataSet)
	{
	    Z_owa = ClassificationDataSet.comineAllBut((List<ClassificationDataSet>)Z_owa_splits, -1);
	    LogisticRegressionDCD lr = new LogisticRegressionDCD();
	    lr.setUseBias(false);
	    Z_model = lr;
	    
//	    if(estimate_cv_scores)
//	    {
//		System.out.println("CV Est:");
//		for(int i = 0; i < scores_c.size(); i++)
//		{
//		    System.out.println(scores_c.get(i).getName() + " : " + scores_stats.get(i).getMean());
//		}
//	    }
	}
	else
	{
	    Z_owa = RegressionDataSet.comineAllBut((List<RegressionDataSet>)Z_owa_splits, -1);
	    DCDs dcd = new DCDs(); 
	    dcd.setUseBias(false);
	    Z_model = dcd;
	}

//	System.out.println("Final model features & sample size " + Z_owa.getNumFeatures() + " " + Z_owa.size());

	GridSearch rs = new GridSearch((Classifier)Z_model, 5); //lazy but OK b/c DCD is also a classifier.
	rs.setUseWarmStarts(false);//B/c we will do parallel run
	rs.autoAddParameters(Z_owa, 9);
	rs.setTrainModelsInParallel(true);
	rs.setTrainFinalModel(true);
	if(Z_owa instanceof ClassificationDataSet)
	    rs.train((ClassificationDataSet) Z_owa, parallel);
	else
	    rs.train((RegressionDataSet) Z_owa, parallel);

	SimpleWeightVectorModel weight_model = (SimpleWeightVectorModel) rs.getTrainedClassifier();

	Vec w_final = new DenseVector(d);
	Vec b_final = new DenseVector(1);
	accumulateUpdates(m, weight_model, w_final, b_final, W, b);

	this.w = w_final;
	this.bias = b_final.get(0);
    }

    /**
     * 
     * @param m the number of models being combined
     * @param w_final the location to store the final averaged weight vector
     * @param w_i_weights_source the aggregating model used to determine how much each of the m sub-models contribute to the final answer
     * @param W matrix of all m model's weights 
     * @param b_final the location to store the final averaged bias term
     * @param b the vector of all m model's bias terms 
     */
    private void accumulateUpdates(final int m, SimpleWeightVectorModel w_i_weights_source, Vec w_final, Vec b_final, Matrix W, Vec b)
    {
	Vec w_i_weights = w_i_weights_source.getRawWeight(0).clone();
	if(w_i_weights.min() >= 0)
	    w_i_weights.mutableDivide(w_i_weights.sum());
	
	for(int i = 0; i < m; i++)
	{
	    w_final.mutableAdd(w_i_weights.get(i), W.getRowView(i));
	    b_final.increment(0, w_i_weights.get(i) * b.get(i));
	}
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
	return LogisticLoss.classify(w.dot(data.getNumericalValues())+bias);
    }

    @Override
    public void train(ClassificationDataSet dataSet, boolean parallel)
    {
	Classifier c_base = (Classifier) base_learner;
	if(!parallel)//why are you doing this?
	{
	    c_base.train(dataSet);
	    this.w = base_learner.getRawWeight();
	    this.bias = base_learner.getBias();
	    return;
	}
	//OK, parallel time! 
	trainWork(-1, dataSet, parallel, null);
    }


    @Override
    public boolean supportsWeightedData()
    {
	if (base_learner instanceof Classifier)
	    return ((Classifier)base_learner).supportsWeightedData();
	else
	    return ((Regressor)base_learner).supportsWeightedData();
    }

    @Override
    public double regress(DataPoint data)
    {
	return w.dot(data.getNumericalValues())+bias;
    }

    @Override
    public void train(RegressionDataSet dataSet, boolean parallel)
    {
	Regressor r_base = (Regressor) base_learner;
	if(!parallel)//why are you doing this?
	{
	    r_base.train(dataSet);
	    this.w = base_learner.getRawWeight();
	    this.bias = base_learner.getBias();
	    return;
	}
	
	trainWork(-1, dataSet, parallel, null);
    }

    @Override
    public Vec getRawWeight()
    {
	return w;
    }

    @Override
    public double getBias()
    {
	return bias;
    }

    @Override
    public OWA clone()
    {
	return new OWA(this);
    }

    @Override
    public boolean warmFromSameDataOnly()
    {
	if( base_learner instanceof WarmClassifier)
	    return ((WarmClassifier)base_learner).warmFromSameDataOnly();
	else if( base_learner instanceof WarmRegressor)
	    return ((WarmRegressor)base_learner).warmFromSameDataOnly();
	else
	    return false;
    }

    @Override
    public void train(ClassificationDataSet dataSet, Classifier warmSolution, boolean parallel)
    {
	Classifier c_base = (Classifier) base_learner;
	if(!parallel)//why are you doing this?
	{
	    c_base.train(dataSet);
	    this.w = base_learner.getRawWeight();
	    this.bias = base_learner.getBias();
	    return;
	}
	//OK, parallel time! 
	trainWork(-1, dataSet, parallel, warmSolution);
    }
    
}
