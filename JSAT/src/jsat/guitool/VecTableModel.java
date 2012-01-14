

package jsat.guitool;

import java.util.List;
import javax.swing.table.AbstractTableModel;
import jsat.DataSet;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class VecTableModel extends AbstractTableModel
{
    private final int rows;
    private final int columns;
    private final DataSet dataSet;

    public VecTableModel(DataSet dataSet)
    {
        this.dataSet = dataSet;
        this.rows = dataSet.getSampleSize();
        this.columns = dataSet.getNumNumericalVars() + dataSet.getNumCategoricalVars();
    }

    public int getRowCount()
    {
        return rows;
    }

    public int getColumnCount()
    {
        return columns;
    }
    
    @Override
    public String getColumnName(int column)
    {
        if(column < dataSet.getNumCategoricalVars())
            return dataSet.getCategoryName(column);
        else
            return dataSet.getNumericName(column - dataSet.getNumCategoricalVars());
    }

    public Object getValueAt(int rowIndex, int columnIndex)
    {
        if(columnIndex < dataSet.getNumCategoricalVars())
            return dataSet.getCategories()[columnIndex].catName(dataSet.getDataPoint(rowIndex).getCategoricalValue(columnIndex));
        else 
            return dataSet.getDataPoint(rowIndex).getNumericalValues().get(columnIndex - dataSet.getNumCategoricalVars());
    }

    @Override
    public void setValueAt(Object aValue, int rowIndex, int columnIndex)
    {
        if(aValue instanceof Double)
            dataSet.getDataPoint(rowIndex).getNumericalValues().set(columnIndex, (Double) aValue);
        else if(aValue instanceof String)
            dataSet.getDataPoint(rowIndex).getNumericalValues().set(columnIndex, Double.parseDouble((String)aValue));
    }




}
