package jsat.distributions;


import jsat.linear.Vec;

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
		throw new UnsupportedOperationException("Not yet implemented");
	}

//	@Override
//	public double invCdf(double p) {
//		if (p >= 1) {
//			return value;
//		} else {
//			return value - 0.00001;
//		}
//	}
//	@Override
//	public double min() {
//		return value - 0.1;
//	}
//
//	@Override
//	public double max() {
//		return value + 0.1;
//	}
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
		// throw new IllegalStateException("Not yet implemented");

	}

	@Override
	public double[] getCurrentVariableValues() {
		return new double[] { value };
		// throw new IllegalStateException("Not yet implemented");

	}

	@Override
	public void setVariable(String var, double value) {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public Distribution clone() {
		return new SingleValueDistribution(this.value);
	}

	@Override
	public void setUsingData(Vec data) {
		throw new UnsupportedOperationException("Not yet implemented");
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
		throw new UnsupportedOperationException("Not yet implemented");
	}

}
