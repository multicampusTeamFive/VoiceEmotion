from django.db import models

# Create your models here.
class MusicUrl(models.Model):
    id = models.AutoField(primary_key=True)
    category = models.CharField(max_length=20, null=True)
    url = models.CharField(max_length=255, null=True)
    title = models.CharField(max_length=255, null=True)
    class Meta:
        managed = False
        db_table = 'MusicUrl'
