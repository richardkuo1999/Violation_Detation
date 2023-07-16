import folium
from folium.plugins import MarkerCluster

class Map:
     def __init__(self) -> None:
          self.map = folium.Map()

          ViolationParkinggroup = folium.FeatureGroup(name='Violation', control=True)
          self.ViolationParkingMarker=MarkerCluster().add_to(ViolationParkinggroup)

          moveroup = folium.FeatureGroup(name='Move', control=True)
          self.MoveMarker=MarkerCluster().add_to(moveroup)

     def add_Marker(self, location, popup, isViolation,):
          addTo = self.ViolationParkingMarker if isViolation else self.MoveMarker

          folium.Marker(location=location,
                    popup=popup).add_to(addTo)
     
     def map_save(self, save_dir):
          self.map.add_child(self.ViolationParkingMarker)    
          self.map.add_child(self.MoveMarker)    
          folium.LayerControl().add_to(self.map)
          self.map.save(save_dir)