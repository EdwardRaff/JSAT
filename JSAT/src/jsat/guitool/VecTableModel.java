

package jsat.guitool;

import java.util.List;
import javax.swing.table.AbstractTableModel;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class VecTableModel extends AbstractTableModel
{
    private final int rows;
    private final int columns;
    private final String headers[];
    private final List<Vec> data;

    public VecTableModel(List<Vec> data, String[] headers)
    {
        this.data = data;
        this.rows = data.get(0).length();
        this.columns = headers.length;
        this.headers = headers;
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
        return headers[column];
    }

    public Object getValueAt(int rowIndex, int columnIndex)
    {
        return data.get(columnIndex).get(rowIndex);
    }

    @Override
    public void setValueAt(Object aValue, int rowIndex, int columnIndex)
    {
        if(aValue instanceof Double)
            data.get(columnIndex).set(rowIndex, (Double)aValue);
        else if(aValue instanceof String)
            data.get(columnIndex).set(rowIndex, Double.parseDouble((String)aValue)); 
    }




}
