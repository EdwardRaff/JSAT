package jsat.distributions;


import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.classifiers.bayesian.MultivariateNormals;
import jsat.linear.Vec;
import jsat.utils.Pair;

public class SingleValueDistribution extends Distribution {

	/**
	 * 
	 */
	private static final long serialVersionUID = 557528557730663203L;
	private double value;

	public SingleValueDistribution(double value) {
		this.value = value;
	}

	@Override
	public double pdf(double x) {
		if (x== value) {
			return 1;
		} else {
			return 0;
		}
	}

	@Override
	public double cdf(double x) {
		if(x >= value){
			return 1;
		}else{
			return 0;
		}
	}

	@Override
	public double invCdf(double p) {
		return value;
	}
	@Override
	public double min() {
		return value;
	}

	@Override
	public double max() {
		return value;
	}



	@Override
	public String getDescriptiveName() {
		return getDistributionName() + "(value=" + value+")";
	}

	@Override
	public String getDistributionName() {
		return "SingleValueDistribution";
	}

	@Override
	public String[] getVariables() {
		return new String[] { "value" };

	}

	@Override
	public double[] getCurrentVariableValues() {
		return new double[] { value };

	}

	@Override
	public void setVariable(String var, double value) {
		if(var.equals("value")){
			this.value = value;
		}
	}

	@Override
	public Distribution clone() {
		return new SingleValueDistribution(this.value);
	}

	@Override
	public void setUsingData(Vec data) {
		Pair<Boolean, Double> sameValues = DistributionSearch.checkForDifferentValues(data);
		if(sameValues.getFirstItem()){
			value = sameValues.getSecondItem();
		}else{
            Logger.getLogger(SingleValueDistribution.class.getName()).log(Level.WARNING,"Trying to use a SingleValueDistribution with data that contains more than one value.");
		}
	}

	@Override
	public double mean() {
		return value;
	}

	@Override
	public double mode() {
		return value;
	}

	@Override
	public double variance() {
		return 0;
	}

	@Override
	public double skewness() {
		return Double.NaN;
	}

}
